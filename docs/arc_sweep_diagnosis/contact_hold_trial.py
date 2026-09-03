"""Direction 2, done at the frame where it matters: copy the parallel run's pose at the contact
event, arrive there smoothly, hold until the arm has actually converged, then carry on.

Why a third attempt. The previous `snap_to_reference` (fix_trial.py, group D in EXPERIMENT_LEDGER.md)
retargeted the whole frozen tail by `delta = ref_contact_pose - last_target_of_subtask`. Those two
poses belong to different moments: the subtask ends `subtask_term_offset_range = (10, 20)` frames
AFTER the gripper transition (data_generator.py randomize_subtask_boundaries adds the offset to the
end index), and the source arm moves several centimetres in those frames. So the tail was pushed
along a vector that mixed "where the reference was at contact" with "where the source arm went
after contact", the ramp had only partly built up by the contact frame, and the hold converged the
arm onto the reference's contact pose after the gripper had already acted. Measured result: the
corrected arm was 1.59 cm from the reference at contact, the uncorrected one 0.69 cm.

What this version does instead. Inside each subtask's pose sequence the contact command frame `c`
is the first frame whose gripper action differs from the segment's first one (each segment holds
exactly one transition, 10-20 frames before its end). Three modes, via CONTACT_FIX:

  none         arc only. Control.
  hold_target  insert HOLD noise-free frames before frame c that hold the nominal target P[c]. The
               source human was at rest on that target when the gripper acted (|target-achieved|
               median 0.00 cm at both grasp and release across the 10 demos), so this lets the
               generated arm arrive where the source arm was, without any reference run.
  snap_ref     the same hold, but at the reference run's achieved end-effector position at its
               contact event (REF_TABLE from build_contact_table.py, matched by initial cube
               layout). The target is ramped onto that position over RAMP frames before c and
               ramped back over RAMP frames after, so there is no jump in either direction. This is
               the "parallel MimicGen, copy its pose, transition smoothly" proposal.

Only the translation is touched; orientation stays the source's. Nothing in the installed tree is
modified: the two hooks are the same monkeypatches fix_trial.py used. Per-episode reseeding (RESEED=1)
pins scene k to the same layout in every run so the reference lookup and paired analysis work.

A hits/misses/applied counter is written to HITS_FILE after every lookup, because this harness's
env.close() takes the process down before main() resumes and a silent no-op would otherwise be
indistinguishable from a null result (three group-D runs were lost that way).
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-Mimic-Perturbed-v0")
parser.add_argument("--attempts", type=int, default=300)
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--input_file", type=str, default="/home/pk/IsaacLab/datasets/annotated_dataset.hdf5")
parser.add_argument("--output_file", type=str, required=True)
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

MODE = os.environ.get("CONTACT_FIX", "none")
HOLD = int(os.environ.get("HOLD", "20"))
RAMP = int(os.environ.get("RAMP", "10"))
REF_TABLE = os.environ.get("REF_TABLE", "")
HITS_FILE = os.environ.get("HITS_FILE", "")
CUBES = ("cube_1", "cube_2", "cube_3")
assert MODE in ("none", "hold_target", "snap_ref"), MODE


def install_contact_fix():
    import isaaclab_mimic.datagen.data_generator as dg
    from isaaclab_mimic.datagen.waypoint import WaypointSequence

    table_keys = table_poses = None
    if MODE == "snap_ref":
        blob = np.load(REF_TABLE)
        table_keys, table_poses = blob["keys"], blob["poses"]
        print(f"[contact] reference table: {len(table_keys)} episodes x {table_poses.shape[1]} contact poses")

    orig_method = dg.DataGenerator.generate_eef_subtask_trajectory
    orig_from_poses = WaypointSequence.from_poses.__func__
    state = {
        "in_subtask": False, "env": None, "subtask": 0, "row": {},
        "hits": 0, "misses": 0, "applied": 0, "no_transition": 0, "nearest": float("nan"),
    }

    def write_counters():
        if HITS_FILE:
            with open(HITS_FILE, "w") as fh:
                fh.write(
                    f"mode={MODE} hits={state['hits']} misses={state['misses']} applied={state['applied']} "
                    f"no_transition={state['no_transition']} last_nearest={state['nearest']:.6f}\n"
                )

    def wrapped_method(self, env_id, eef_name, subtask_ind, *a, **kw):
        if subtask_ind == 0 and table_keys is not None:
            # Match by what is on the table, in the env-local frame the recorder writes (root_pos_w
            # carries the environment's grid origin; subtracting it is what fixed the d2c runs).
            origin = self.env.scene.env_origins[env_id]
            layout = np.concatenate(
                [(self.env.scene[c].data.root_pos_w[env_id] - origin).detach().cpu().numpy() for c in CUBES]
            )
            d = np.linalg.norm(table_keys - layout, axis=1)
            i = int(np.argmin(d))
            # Reseeding reproduces a layout to ~0.2 mm median / 1.4 mm p90; the nearest different
            # layout is never closer than 23 mm. 2 mm accepts ~97% with no ambiguous case.
            state["row"][env_id] = i if d[i] < 2e-3 else None
            state["nearest"] = float(d[i])
            if state["row"][env_id] is None:
                state["misses"] += 1
            else:
                state["hits"] += 1
            write_counters()
        state["env"] = env_id
        state["subtask"] = subtask_ind
        state["in_subtask"] = True
        try:
            return orig_method(self, env_id, eef_name, subtask_ind, *a, **kw)
        finally:
            state["in_subtask"] = False

    def fixed_from_poses(cls, poses, gripper_actions, action_noise):
        # Only the subtask segment built inside generate_eef_subtask_trajectory is touched; the
        # one-pose init sequence and the interpolation frames built by merge() pass through.
        if not state["in_subtask"] or poses.shape[0] < 2 or MODE == "none":
            return orig_from_poses(cls, poses, gripper_actions, action_noise)
        T = poses.shape[0]
        changed = torch.nonzero((gripper_actions != gripper_actions[0]).any(dim=-1)).flatten()
        if len(changed) == 0 or int(changed[0]) == 0:
            state["no_transition"] += 1
            write_counters()
            return orig_from_poses(cls, poses, gripper_actions, action_noise)
        c = int(changed[0])

        if torch.is_tensor(action_noise):
            noise = action_noise.reshape(-1, 1).clone().to(dtype=torch.float32)
        else:
            noise = torch.full((T, 1), float(action_noise), dtype=torch.float32)

        hold_pos = poses[c, :3, 3].clone()
        if MODE == "snap_ref":
            row = state["row"].get(state["env"])
            k = state["subtask"]
            if row is not None and k < table_poses.shape[1]:
                hold_pos = torch.as_tensor(table_poses[row, k], dtype=poses.dtype, device=poses.device)
            # else: no reference for this scene -> degrade to hold_target for this subtask
        offset = hold_pos - poses[c, :3, 3]

        P = poses.clone()
        if float(offset.norm()) > 0.0:
            m_in = min(RAMP, c)
            if m_in > 0:
                s = torch.linspace(0.0, 1.0, m_in + 1, device=P.device, dtype=P.dtype)[1:]
                P[c - m_in:c, :3, 3] += s[:, None] * offset[None, :]
            m_out = min(RAMP, T - c)
            if m_out > 0:
                s2 = torch.linspace(1.0, 0.0, m_out + 1, device=P.device, dtype=P.dtype)[:-1]
                P[c:c + m_out, :3, 3] += s2[:, None] * offset[None, :]

        hold = P[c:c + 1].expand(HOLD, -1, -1).clone()
        hold[:, :3, 3] = hold_pos[None, :]
        new_poses = torch.cat([P[:c], hold, P[c:]], dim=0)
        new_grip = torch.cat([gripper_actions[:c], gripper_actions[c - 1:c].expand(HOLD, -1), gripper_actions[c:]], dim=0)
        new_noise = torch.cat([noise[:c], torch.zeros((HOLD, 1), dtype=noise.dtype), noise[c:]], dim=0)
        state["applied"] += 1
        write_counters()
        return orig_from_poses(cls, new_poses, new_grip, new_noise)

    dg.DataGenerator.generate_eef_subtask_trajectory = wrapped_method
    WaypointSequence.from_poses = classmethod(fixed_from_poses)
    return state


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

    install_contact_fix()
    print(f"[contact] mode={MODE} hold={HOLD} ramp={RAMP} table={REF_TABLE or '-'}")
    print(f"[cfg] arc_std={os.environ.get('PERTURB_ARC_STD')} attempts={args_cli.attempts}"
          f" guarantee={env_cfg.datagen_config.generation_guarantee} seed={env_cfg.datagen_config.seed}")

    env = gym.make(env_name, cfg=env_cfg).unwrapped

    if os.environ.get("RESEED"):
        orig_reset = env.reset
        reset_count = {"n": 0}

        def seeded_reset(*a, **kw):
            seed = 1_000_000 + reset_count["n"]
            reset_count["n"] += 1
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))
            random.seed(seed)
            return orig_reset(*a, **kw)

        env.reset = seeded_reset
        print("[contact] deterministic per-episode reseeding enabled")
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
    print(f"\nRESULT mode={MODE} arc={os.environ.get('PERTURB_ARC_STD')}"
          f" successes={gen_mod.num_success} attempts={gen_mod.num_attempts} rate={rate:.2f}%")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
