"""Closed-loop validation harness for two Isaac Lab Mimic defects.

Runs the real generation loop on a stock upstream task. Nothing in the installed tree is modified:
the harness injects the candidate fixes into the objects `setup_env_config` hands back, which is the
same path the data generator reads them from.

Two defects under test.

1. `DataGenConfig.max_num_failures` is documented as "Maximum number of failures allowed before
   stopping generation" and is set to 25 by eighteen shipped env configs, but nothing ever reads it.
   The only termination is "enough successes" (or "enough attempts" when generation_guarantee is
   False), so a task with a low success rate retries without bound.
   FIX_MAX_FAILURES=1 swaps in an env_loop that honours the field.

2. `cubes_stacked` tests an instantaneous geometric configuration with no requirement that the
   cubes be at rest, so a cube released above its target satisfies it for a frame or two on the way
   down and the whole episode is marked a success.
   FIX_AT_REST=<v> requires both movable cubes to be below v m/s as well as in position.

Environment variables:
  FIX_MAX_FAILURES=1        honour datagen_config.max_num_failures
  FIX_AT_REST=0.01          add the at-rest requirement at this threshold (m/s)
  MAX_NUM_FAILURES=<n>      override the config value, for a shorter demonstration
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate Isaac Lab Mimic generation defects.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--generation_num_trials", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, default="./datasets/validate_out.hdf5")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import contextlib
import functools
import random

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.datagen import generation as gen_mod
from isaaclab_mimic.datagen.generation import setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401

FIX_MAX_FAILURES = os.environ.get("FIX_MAX_FAILURES", "0") == "1"
FIX_AT_REST = float(os.environ.get("FIX_AT_REST", "0") or 0)
MAX_NUM_FAILURES_OVERRIDE = os.environ.get("MAX_NUM_FAILURES")


def env_loop_with_failure_cap(env, env_reset_queue, env_action_queue, shared_datagen_info_pool, asyncio_event_loop):
    """The shipped env_loop with the documented max_num_failures check actually wired in.

    Everything except the marked block is a copy of isaaclab_mimic.datagen.generation.env_loop.
    """
    env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
    prev_num_attempts = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
            while env_action_queue.qsize() != env.num_envs:
                asyncio_event_loop.run_until_complete(asyncio.sleep(0))
                while not env_reset_queue.empty():
                    env_id_tensor[0] = env_reset_queue.get_nowait()
                    env.reset(env_ids=env_id_tensor)
                    env_reset_queue.task_done()

            actions = torch.zeros(env.action_space.shape)
            for _ in range(env.num_envs):
                env_id, action = asyncio_event_loop.run_until_complete(env_action_queue.get())
                actions[env_id] = action

            env.step(actions)

            for _ in range(env.num_envs):
                env_action_queue.task_done()

            if prev_num_attempts != gen_mod.num_attempts:
                prev_num_attempts = gen_mod.num_attempts
                rate = 100 * gen_mod.num_success / gen_mod.num_attempts if gen_mod.num_attempts else 0.0
                print(f"\n{'*' * 50}\n{gen_mod.num_success}/{gen_mod.num_attempts} ({rate:.1f}%) successful\n{'*' * 50}")

                generation_guarantee = env.cfg.datagen_config.generation_guarantee
                generation_num_trials = env.cfg.datagen_config.generation_num_trials
                check_val = gen_mod.num_success if generation_guarantee else gen_mod.num_attempts
                if check_val >= generation_num_trials:
                    print(f"Reached {generation_num_trials} successes/attempts. Exiting.")
                    break

                # ---- the fix under test ----
                max_num_failures = env.cfg.datagen_config.max_num_failures
                if max_num_failures is not None and gen_mod.num_failures >= max_num_failures:
                    print(f"[FIX] Reached {gen_mod.num_failures} failures (cap {max_num_failures}). Exiting.")
                    break
                # ---- end fix ----

            if env.sim.is_stopped():
                break

    env.close()


def make_at_rest_criterion(original_func, max_lin_vel: float):
    """Wrap cubes_stacked so a cube that is still moving cannot satisfy it."""

    @functools.wraps(original_func)
    def wrapped(env, **kwargs):
        stacked = original_func(env, **kwargs)
        for name in ("cube_2", "cube_3"):
            cube = env.scene[name]
            speed = torch.linalg.norm(cube.data.root_lin_vel_w, dim=1)
            stacked = torch.logical_and(speed < max_lin_vel, stacked)
        return stacked

    return wrapped


def main():
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task.split(":")[-1] if args_cli.task else None
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
    )

    if MAX_NUM_FAILURES_OVERRIDE is not None:
        env_cfg.datagen_config.max_num_failures = int(MAX_NUM_FAILURES_OVERRIDE)
    print(f"[cfg] max_num_failures = {env_cfg.datagen_config.max_num_failures}")
    print(f"[cfg] generation_guarantee = {env_cfg.datagen_config.generation_guarantee}")
    print(f"[cfg] generation_num_trials = {env_cfg.datagen_config.generation_num_trials}")
    print(f"[cfg] seed = {env_cfg.datagen_config.seed}")

    if FIX_AT_REST > 0:
        success_term.func = make_at_rest_criterion(success_term.func, FIX_AT_REST)
        print(f"[fix] success criterion now requires cubes below {FIX_AT_REST} m/s")

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)
    env.reset()

    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
    )

    loop_fn = env_loop_with_failure_cap if FIX_MAX_FAILURES else gen_mod.env_loop
    print(f"[run] env_loop = {'PATCHED (failure cap honoured)' if FIX_MAX_FAILURES else 'stock'}")

    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        loop_fn(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    finally:
        data_gen_tasks.cancel()
        try:
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error cancelling remaining async tasks: {e}")

    print("\n" + "=" * 60)
    print(f"RESULT successes={gen_mod.num_success} failures={gen_mod.num_failures} attempts={gen_mod.num_attempts}")
    print(f"RESULT max_num_failures={env.cfg.datagen_config.max_num_failures} fix_active={FIX_MAX_FAILURES}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
