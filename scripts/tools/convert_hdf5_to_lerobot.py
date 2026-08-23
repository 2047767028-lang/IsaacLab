"""Convert Isaac Lab / MimicGen robomimic-format HDF5 datasets to LeRobot dataset format
(for training OpenPI / pi0.5).

Only successful demos are expected as input (i.e. point this at a `generated.hdf5` that
already contains only successes, not `generated_failed.hdf5`).

Action target: uses the `actions` field (7-dim: IK-Rel delta pose + binary gripper),
NOT `processed_actions` (8-dim). `processed_actions` is the env-internal, controller-scaled
command actually sent to the physics sim (its first 6 dims are exactly half of `actions`'
first 6 dims - confirmed by inspection) and is not the right imitation-learning target.

State: eef_pos (3) + eef_quat (4) + gripper_pos (2) = 9-dim, concatenated in that order.

Must be run with an environment that has the LeRobot v2.1 dataset API installed
(this repo's `lerobot_v21` conda env), e.g.:
    /home/pk/miniconda3/envs/lerobot_v21/bin/python scripts/tools/convert_hdf5_to_lerobot.py ...

Example:
    python scripts/tools/convert_hdf5_to_lerobot.py \
        --src /home/pk/IsaacLab/datasets/pi05_training_data_v1/baseline/generated.hdf5 \
        --repo-name pi05_training_data_v1_baseline \
        --task "stack the cubes" \
        --max-episodes 5
"""

import argparse
import shutil

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

IMAGE_H = 200
IMAGE_W = 200
STATE_DIM = 9  # eef_pos(3) + eef_quat(4) + gripper_pos(2)
ACTION_DIM = 7  # IK-Rel delta pose (6) + binary gripper (1)
FPS = 20.0  # = 1 / (dt * decimation) = 1 / (0.01 * 5), from data/env_args in the hdf5


def build_state(obs: h5py.Group, t: int) -> np.ndarray:
    return np.concatenate(
        [
            obs["eef_pos"][t],
            obs["eef_quat"][t],
            obs["gripper_pos"][t],
        ]
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="path to generated.hdf5 (successes only)")
    parser.add_argument("--repo-name", required=True, help="output dataset name under HF_LEROBOT_HOME")
    parser.add_argument("--task", default="stack the cubes", help="fixed language instruction for all episodes")
    parser.add_argument("--max-episodes", type=int, default=None, help="limit number of demos (for smoke tests)")
    parser.add_argument("--robot-type", default="franka")
    args = parser.parse_args()

    output_path = HF_LEROBOT_HOME / args.repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_name,
        robot_type=args.robot_type,
        fps=FPS,
        features={
            "image": {
                "dtype": "image",
                "shape": (IMAGE_H, IMAGE_W, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (IMAGE_H, IMAGE_W, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["actions"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    with h5py.File(args.src, "r") as f:
        demo_keys = list(f["data"].keys())
        if args.max_episodes is not None:
            demo_keys = demo_keys[: args.max_episodes]

        for demo_idx, demo_key in enumerate(demo_keys):
            if demo_idx % 20 == 0:
                print(f"PROGRESS: {demo_idx}/{len(demo_keys)} episodes", flush=True)
            demo = f["data"][demo_key]
            if not bool(demo.attrs.get("success", True)):
                continue

            obs = demo["obs"]
            actions = demo["actions"][:]
            table_cam = obs["table_cam"][:]
            wrist_cam = obs["wrist_cam"][:]
            num_frames = actions.shape[0]

            for t in range(num_frames):
                dataset.add_frame(
                    {
                        "image": table_cam[t],
                        "wrist_image": wrist_cam[t],
                        "state": build_state(obs, t),
                        "actions": actions[t].astype(np.float32),
                    },
                    task=args.task,
                )
            dataset.save_episode()

    print(f"Done. Dataset written to: {output_path}")


if __name__ == "__main__":
    main()
