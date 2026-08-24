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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random

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


def rollout(policy, env, success_term, horizon: int, prompt: str) -> bool:
    """Run a single closed-loop rollout, querying the policy server every step.

    Returns:
        Whether the rollout succeeded (per the task's own success termination function).
    """
    obs_dict, _ = env.reset()

    for _ in range(horizon):
        obs = build_observation(obs_dict, prompt)

        # Query the policy server for a chunk of actions; only the first action of the chunk is
        # applied before re-querying with a fresh observation (closed-loop, not open-loop replay).
        action_chunk = policy.infer(obs)["actions"]
        action = np.asarray(action_chunk)[0]
        action_t = torch.from_numpy(action).float().to(device=env.device).view(1, env.action_space.shape[1])

        obs_dict, _, terminated, truncated, _ = env.step(action_t)

        if bool(success_term.func(env, **success_term.params)[0]):
            return True
        elif terminated or truncated:
            return False

    return False


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
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        succeeded = rollout(policy, env, success_term, args_cli.horizon, args_cli.prompt)
        results.append(succeeded)
        print(f"[INFO] Trial {trial}: {succeeded}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
