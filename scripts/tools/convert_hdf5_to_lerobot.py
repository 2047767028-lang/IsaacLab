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

`--src` may be repeated to build one dataset out of several sources, with
`--episodes-per-src` fixing how many episodes each contributes. That combination answers a
different question than either source alone: comparing "N clean demos" against "N/2 clean +
N/2 augmented" isolates whether the augmented demos are individually worth as much as clean
ones, without the dataset-size confound that a simple union would introduce. Episodes are
sampled with `--sample-seed` (not taken from the front, in case generation order correlates
with anything) and interleaved round-robin across sources:

    python scripts/tools/convert_hdf5_to_lerobot.py \
        --src .../baseline/generated.hdf5 --src .../arc_1p2cm/generated.hdf5 \
        --episodes-per-src 190 --repo-name pi05_lerobot_mixed_half
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


def plan_episodes(srcs: list[str], episodes_per_src: int | None, max_episodes: int | None, seed: int):
    """Decide which (source, demo_key) pairs go into the dataset, and in what order.

    Returns the interleaved plan plus the per-source selections, so the caller can report exactly
    what went in -- with several sources the composition is the whole point of the run.
    """
    picked = []
    for src in srcs:
        with h5py.File(src, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            keys = [k for k in keys if bool(f["data"][k].attrs.get("success", True))]
        n = episodes_per_src if episodes_per_src is not None else (max_episodes or len(keys))
        if n > len(keys):
            raise SystemExit(f"{src} only has {len(keys)} successful demos, cannot take {n}")
        if n < len(keys):
            rng = np.random.default_rng(seed)
            keys = [keys[i] for i in sorted(rng.choice(len(keys), size=n, replace=False))]
        picked.append((src, keys))

    plan = []
    for i in range(max(len(k) for _, k in picked)):
        for src, keys in picked:
            if i < len(keys):
                plan.append((src, keys[i]))
    return plan, picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, action="append",
                        help="path to generated.hdf5 (successes only); repeat to mix sources")
    parser.add_argument("--repo-name", required=True, help="output dataset name under HF_LEROBOT_HOME")
    parser.add_argument("--task", default="stack the cubes", help="fixed language instruction for all episodes")
    parser.add_argument("--max-episodes", type=int, default=None, help="limit number of demos (for smoke tests)")
    parser.add_argument("--episodes-per-src", type=int, default=None,
                        help="how many episodes to take from EACH --src (required when mixing)")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="seed for choosing which episodes to take when taking a subset")
    parser.add_argument("--robot-type", default="franka")
    args = parser.parse_args()

    if len(args.src) > 1 and args.episodes_per_src is None:
        raise SystemExit("--episodes-per-src is required when passing more than one --src")

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

    plan, picked = plan_episodes(args.src, args.episodes_per_src, args.max_episodes, args.sample_seed)
    for src, keys in picked:
        print(f"SOURCE {src}: taking {len(keys)} episodes, first few = {keys[:5]}", flush=True)
    print(f"TOTAL {len(plan)} episodes, interleaved across {len(picked)} source(s)", flush=True)

    handles = {src: h5py.File(src, "r") for src in args.src}
    try:
        for idx, (src, demo_key) in enumerate(plan):
            if idx % 20 == 0:
                print(f"PROGRESS: {idx}/{len(plan)} episodes", flush=True)
            demo = handles[src]["data"][demo_key]

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
    finally:
        for h in handles.values():
            h.close()

    print(f"Done. Dataset written to: {output_path}")


if __name__ == "__main__":
    main()
