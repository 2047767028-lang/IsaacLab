"""Try the two cheap remedies against the arc penalty, on the same task the v2 sweep used.

Both are injected into the config `setup_env_config` returns, so nothing in the installed tree is
modified and the generation loop is the stock one.

  DWELL=<n>   direction 3. SubTaskConfig.num_fixed_steps, currently 0, inserts a segment of constant
              target pose before each subtask's trajectory. A static target is the one condition
              under which the Cartesian loop can actually converge -- while the target keeps moving,
              two runs settle at different steady-state lags and the difference between them decays
              0.76% per frame.

  SCALE=<s>   direction 2. DifferentialInverseKinematicsActionCfg.scale, currently 0.5, is how much
              of the position error the controller commands per step. The arm executes about 20% of
              what it is commanded, so raising this is the direct way to ask for a faster return to
              the unperturbed path.

Fixed-attempts mode, so every run costs the same number of attempts regardless of how well it does.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-Mimic-Perturbed-v0")
parser.add_argument("--attempts", type=int, default=300)
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--input_file", type=str, default="/home/pk/IsaacLab/datasets/annotated_dataset.hdf5")
parser.add_argument("--output_file", type=str, default="/home/pk/.claude/jobs/10fee75c/tmp/out/fix.hdf5")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import random

import gymnasium as gym
import numpy as np
import torch

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.datagen import generation as gen_mod
from isaaclab_mimic.datagen.generation import setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401

DWELL = int(os.environ.get("DWELL", "0"))
SCALE = float(os.environ.get("SCALE", "0"))


def main():
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    env_name = args_cli.task.split(":")[-1] or get_env_name_from_dataset(args_cli.input_file)

    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.attempts,
    )

    if DWELL > 0:
        for eef, subtasks in env_cfg.subtask_configs.items():
            for st in subtasks:
                st.num_fixed_steps = DWELL
        print(f"[fix] num_fixed_steps = {DWELL} on every subtask")
    if SCALE > 0:
        old = env_cfg.actions.arm_action.scale
        env_cfg.actions.arm_action.scale = SCALE
        print(f"[fix] arm_action.scale {old} -> {SCALE}")

    print(f"[cfg] arc_std={os.environ.get('PERTURB_ARC_STD')} attempts={args_cli.attempts}"
          f" guarantee={env_cfg.datagen_config.generation_guarantee} seed={env_cfg.datagen_config.seed}")

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)
    env.reset()

    comp = setup_async_generation(
        env=env, num_envs=args_cli.num_envs, input_file=args_cli.input_file, success_term=success_term
    )
    try:
        tasks = asyncio.ensure_future(asyncio.gather(*comp["tasks"]))
        gen_mod.env_loop(env, comp["reset_queue"], comp["action_queue"], comp["info_pool"], comp["event_loop"])
    except asyncio.CancelledError:
        pass
    finally:
        tasks.cancel()
        try:
            comp["event_loop"].run_until_complete(tasks)
        except Exception:
            pass

    rate = 100 * gen_mod.num_success / gen_mod.num_attempts if gen_mod.num_attempts else 0.0
    print(f"\nRESULT dwell={DWELL} scale={SCALE or 'default'} arc={os.environ.get('PERTURB_ARC_STD')}"
          f" successes={gen_mod.num_success} attempts={gen_mod.num_attempts} rate={rate:.2f}%")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
