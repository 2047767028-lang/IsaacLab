# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a pi0.5 (openpi) policy served over websocket in an Isaac Lab environment.

This is the openpi analogue of scripts/imitation_learning/robomimic/play.py: instead of loading a
robomimic checkpoint locally, it connects to an already-running openpi `serve_policy.py` server and
queries it for actions every step (closed-loop). No perturbation is applied here on purpose -- this
is meant as the first, standard sanity check that the "policy server <-> Isaac Sim rollout" bridge
itself works, before layering any evaluation-time perturbation sweep on top.

Args:
    task: Name of the environment (defaults to the plain, non-Mimic, non-Perturbed visuomotor task).
    policy_host / policy_port: Where the openpi websocket policy server is listening.
    prompt: Language instruction sent to the policy on every query.
    horizon: Max steps per rollout before declaring failure.
    num_rollouts: Number of rollouts to run.
    seed: Random seed.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an openpi policy server for an Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0", help="Name of the task."
)
parser.add_argument("--policy_host", type=str, default="localhost", help="Host of the openpi policy server.")
parser.add_argument("--policy_port", type=int, default=8000, help="Port of the openpi policy server.")
parser.add_argument(
    "--prompt", type=str, default="stack the cubes", help="Language instruction sent to the policy."
)
parser.add_argument("--horizon", type=int, default=400, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--results_file",
    type=str,
    default=None,
    help="Optional path to write per-rollout results + summary as JSON (rewritten after every trial).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from openpi_client import websocket_client_policy as _websocket_client_policy

from isaaclab_tasks.utils import parse_env_cfg


def build_observation(obs_dict: dict, prompt: str) -> dict:
    """Convert an Isaac Lab policy-group observation dict into the libero-style dict openpi expects."""
    policy_obs = obs_dict["policy"]

    eef_pos = policy_obs["eef_pos"].squeeze(0).cpu().numpy()
    eef_quat = policy_obs["eef_quat"].squeeze(0).cpu().numpy()
    gripper_pos = policy_obs["gripper_pos"].squeeze(0).cpu().numpy()
    state = np.concatenate([eef_pos, eef_quat, gripper_pos]).astype(np.float32)

    # mdp.image(..., normalize=False) returns raw uint8 HWC tensors -- no rescaling needed here,
    # openpi's LiberoInputs transform handles uint8 HWC directly.
    table_image = policy_obs["table_cam"].squeeze(0).cpu().numpy()
    wrist_image = policy_obs["wrist_cam"].squeeze(0).cpu().numpy()

    return {
        "observation/state": state,
        "observation/image": table_image,
        "observation/wrist_image": wrist_image,
        "prompt": prompt,
    }


def rollout(policy, env, success_term, horizon: int, prompt: str) -> dict:
    """Run a single closed-loop rollout, querying the policy server every step.

    Returns:
        A dict with the outcome plus per-rollout timing/step diagnostics. Step counts matter as a
        sanity check: a "success" that lands in a handful of steps would mean the success term is
        being satisfied trivially rather than by actually stacking the cubes.
    """
    obs_dict, _ = env.reset()

    steps = 0
    infer_seconds = 0.0
    wall_start = time.time()
    outcome = "horizon_exhausted"

    for _ in range(horizon):
        obs = build_observation(obs_dict, prompt)

        # Query the policy server for a chunk of actions; only the first action of the chunk is
        # applied before re-querying with a fresh observation (closed-loop, not open-loop replay).
        infer_start = time.time()
        action_chunk = policy.infer(obs)["actions"]
        infer_seconds += time.time() - infer_start

        action = np.asarray(action_chunk)[0]
        action_t = torch.from_numpy(action).float().to(device=env.device).view(1, env.action_space.shape[1])

        obs_dict, _, terminated, truncated, _ = env.step(action_t)
        steps += 1

        if bool(success_term.func(env, **success_term.params)[0]):
            outcome = "success"
            break
        elif terminated or truncated:
            outcome = "terminated"
            break

    return {
        "success": outcome == "success",
        "outcome": outcome,
        "steps": steps,
        "wall_seconds": round(time.time() - wall_start, 2),
        "infer_seconds": round(infer_seconds, 2),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- behaves sensibly near 0 and 1, unlike the normal approximation."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def main():
    """Evaluate an openpi policy server against an Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Keep observation terms as a dict (not concatenated) so we can pick out eef_pos/eef_quat/
    # gripper_pos/table_cam/wrist_cam by name.
    env_cfg.observations.policy.concatenate_terms = False

    # We drive the episode horizon ourselves.
    env_cfg.terminations.time_out = None

    # Disable recorder -- this is an eval rollout, not a demo-generation run.
    env_cfg.recorders = None

    # Extract success checking function so we can check it every step without it auto-terminating.
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Connect to the already-running openpi policy server.
    policy = _websocket_client_policy.WebsocketClientPolicy(host=args_cli.policy_host, port=args_cli.policy_port)
    print(f"[INFO] Connected to policy server. Server metadata: {policy.get_server_metadata()}")

    # Run policy
    records = []
    for trial in range(args_cli.num_rollouts):
        record = rollout(policy, env, success_term, args_cli.horizon, args_cli.prompt)
        record["trial"] = trial
        records.append(record)

        n_success = sum(r["success"] for r in records)
        lo, hi = wilson_interval(n_success, len(records))
        print(
            f"[INFO] Trial {trial}: {record['outcome']} "
            f"(steps={record['steps']}, {record['wall_seconds']}s, infer={record['infer_seconds']}s) "
            f"| running {n_success}/{len(records)} = {n_success / len(records):.3f} "
            f"[95% CI {lo:.3f}-{hi:.3f}]",
            flush=True,
        )

        # Write incrementally so a long run stays inspectable (and salvageable) while in flight.
        if args_cli.results_file:
            summary = build_summary(records)
            Path(args_cli.results_file).parent.mkdir(parents=True, exist_ok=True)
            with open(args_cli.results_file, "w") as f:
                json.dump(summary, f, indent=2)

    summary = build_summary(records)
    print("\n" + "=" * 70)
    print(f"Task:            {args_cli.task}")
    print(f"Seed:            {args_cli.seed}")
    print(f"Successful:      {summary['n_success']} / {summary['n_trials']}")
    print(f"Success rate:    {summary['success_rate']:.4f}")
    print(f"95% CI (Wilson): {summary['ci_low']:.4f} - {summary['ci_high']:.4f}  (+-{summary['ci_halfwidth']:.4f})")
    print(f"Outcomes:        {summary['outcome_counts']}")
    print(f"Mean steps:      success={summary['mean_steps_success']}, failure={summary['mean_steps_failure']}")
    print(f"Mean wall/roll:  {summary['mean_wall_seconds']}s (inference {summary['infer_fraction']:.1%} of it)")
    print(f"Total wall:      {summary['total_wall_seconds']}s")
    print("=" * 70 + "\n")

    env.close()


def build_summary(records: list[dict]) -> dict:
    """Aggregate per-rollout records into the summary that gets printed and written to disk."""
    n_trials = len(records)
    n_success = sum(r["success"] for r in records)
    lo, hi = wilson_interval(n_success, n_trials)

    succ_steps = [r["steps"] for r in records if r["success"]]
    fail_steps = [r["steps"] for r in records if not r["success"]]
    total_wall = sum(r["wall_seconds"] for r in records)
    total_infer = sum(r["infer_seconds"] for r in records)

    outcome_counts: dict[str, int] = {}
    for r in records:
        outcome_counts[r["outcome"]] = outcome_counts.get(r["outcome"], 0) + 1

    return {
        "task": args_cli.task,
        "seed": args_cli.seed,
        "horizon": args_cli.horizon,
        "prompt": args_cli.prompt,
        "policy_port": args_cli.policy_port,
        "n_trials": n_trials,
        "n_success": n_success,
        "success_rate": n_success / n_trials if n_trials else 0.0,
        "ci_low": lo,
        "ci_high": hi,
        "ci_halfwidth": (hi - lo) / 2,
        "outcome_counts": outcome_counts,
        "mean_steps_success": round(sum(succ_steps) / len(succ_steps), 1) if succ_steps else None,
        "mean_steps_failure": round(sum(fail_steps) / len(fail_steps), 1) if fail_steps else None,
        "mean_wall_seconds": round(total_wall / n_trials, 2) if n_trials else None,
        "total_wall_seconds": round(total_wall, 1),
        "infer_fraction": total_infer / total_wall if total_wall else 0.0,
        "records": records,
    }


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
