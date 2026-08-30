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
SUBTASKS = os.environ.get("ARC_SUBTASKS", "")
"""Comma-separated subtask indices to perturb, e.g. "0". Empty means all of them, as today."""


def gate_arc_to_subtasks(allowed: set[int]):
    """Apply the arc only on the listed subtasks, leaving the rest exactly as the source demo.

    The residual at the contact phase accumulates across subtasks like a random walk -- measured
    0.63, 0.96, 1.01, 1.28 cm at the four contact events for a 3.0 cm arc -- so perturbing k of the
    four should scale it by sqrt(k/4). Perturbing one subtask halves the damage while keeping full
    amplitude where it is applied.

    `_apply_arc_perturbation` does not receive the subtask index, so the enclosing method is wrapped
    to record it first. That method is synchronous, so a module-level slot is safe even though
    generation runs several environments concurrently.

    The original is called either way and its result discarded when gated out, so the random draws
    it makes are consumed identically and the scene sequence stays comparable across runs.
    """
    import isaaclab_mimic.datagen.data_generator as dg

    current = {"ind": None}
    orig_method = dg.DataGenerator.generate_eef_subtask_trajectory
    orig_arc = dg._apply_arc_perturbation

    def wrapped_method(self, env_id, eef_name, subtask_ind, *a, **kw):
        current["ind"] = subtask_ind
        return orig_method(self, env_id, eef_name, subtask_ind, *a, **kw)

    def gated_arc(poses, magnitude, freeze_frac, peak_frac_range=(0.5, 0.5)):
        perturbed = orig_arc(poses, magnitude, freeze_frac, peak_frac_range)
        return perturbed if current["ind"] in allowed else poses

    dg.DataGenerator.generate_eef_subtask_trajectory = wrapped_method
    dg._apply_arc_perturbation = gated_arc


def add_tail_dwell(n: int):
    """Hold the target still for n frames at the END of each subtask, just before contact.

    num_fixed_steps puts its dwell at the START of each subtask, which lands around the grasps and
    never before a placement -- and placement is 66% of the failures. Measured at 3.0 cm: a dwell of
    20 took grasp failures from 16 to 11 and from 31 to 23 while leaving placement failures at 154
    against 155. Raising the controller gain instead fixed grasping harder (16 to 8, 31 to 8) and
    wrecked placement (154 to 216), because a faster arm flicks the cube as it lets go.

    So the dwell wanted is at the other end of the subtask. It is not exposed by any config, but the
    perturbation hook returns the subtask's pose sequence, so repeating its final pose appends the
    dwell there. The gripper-action sequence has to grow to match, which is what the from_poses
    wrapper does.
    """
    import isaaclab_mimic.datagen.data_generator as dg
    from isaaclab_mimic.datagen.waypoint import WaypointSequence

    orig_arc = dg._apply_arc_perturbation
    orig_from_poses = WaypointSequence.from_poses.__func__

    def arc_then_dwell(poses, magnitude, freeze_frac, peak_frac_range=(0.5, 0.5)):
        out = orig_arc(poses, magnitude, freeze_frac, peak_frac_range)
        return torch.cat([out, out[-1:].expand(n, -1, -1)], dim=0)

    def padded_from_poses(cls, poses, gripper_actions, action_noise):
        if len(gripper_actions) < len(poses):
            pad = len(poses) - len(gripper_actions)
            gripper_actions = torch.cat([gripper_actions, gripper_actions[-1:].expand(pad, -1)], dim=0)
            # The appended frames must be noise-free or the dwell does nothing: action_noise is 0.03,
            # i.e. 3 cm per axis, so an arm holding position is shoved around faster than it can
            # settle. The shipped fixed segment gets 0 noise for exactly this reason
            # (apply_noise_during_interpolation is False), and the first version of this patch
            # inherited the subtask's noise instead -- which is why it produced 0.9% of frames within
            # a centimetre of target against the shipped dwell's 2.8%, and no effect on success.
            if not torch.is_tensor(action_noise):
                action_noise = torch.full((len(poses),), float(action_noise))
            action_noise = action_noise.clone()
            action_noise[-pad:] = 0.0
        return orig_from_poses(cls, poses, gripper_actions, action_noise)

    dg._apply_arc_perturbation = arc_then_dwell
    WaypointSequence.from_poses = classmethod(padded_from_poses)


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
    if SUBTASKS:
        allowed = {int(x) for x in SUBTASKS.split(",")}
        gate_arc_to_subtasks(allowed)
        print(f"[fix] arc applied only on subtasks {sorted(allowed)}")
    tail = int(os.environ.get("TAIL_DWELL", "0"))
    if tail > 0:
        add_tail_dwell(tail)
        print(f"[fix] {tail}-frame dwell appended at the end of every subtask")

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
