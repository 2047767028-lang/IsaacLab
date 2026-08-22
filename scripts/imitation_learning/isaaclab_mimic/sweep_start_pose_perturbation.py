# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sweep reset-time start-pose perturbation magnitude (PERTURB_STD) for the
Isaac-Stack-Cube-Franka-IK-Rel-Mimic-Perturbed-v0 env and measure MimicGen
generation success rate at each magnitude.

For the "contact-anchored perturbation augmentation" validation experiment
(docs/接触锚定扰动增广_设计记录.md). Each sigma value is run as a separate
subprocess invocation of generate_dataset.py (Isaac Sim's AppLauncher can only
be started once per process), so this script does not import isaaclab/omni
itself and can run in the plain `isaaclab` conda env without launching the app.

Usage example:
    python scripts/imitation_learning/isaaclab_mimic/sweep_start_pose_perturbation.py \
        --sigmas 0.0,0.02,0.05,0.10,0.20 \
        --trials-per-sigma 50 \
        --num-envs 10 \
        --input-file ./datasets/annotated_dataset.hdf5 \
        --output-dir ./datasets/perturbation_sweep_v1 \
        --results-csv ./docs/perturbation_sweep_v1_results.csv
"""

import argparse
import csv
import os
import subprocess
import time

import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep start-pose perturbation magnitude for MimicGen generation.")
    parser.add_argument("--sigmas", type=str, required=True, help="Comma-separated list of PERTURB_STD values.")
    parser.add_argument(
        "--trials-per-sigma", type=int, required=True, help="Target number of successful demos per sigma value."
    )
    parser.add_argument("--num-envs", type=int, default=10, help="Number of parallel envs for generation.")
    parser.add_argument(
        "--input-file", type=str, default="./datasets/annotated_dataset.hdf5", help="Source annotated dataset."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./datasets/perturbation_sweep_v1", help="Directory for generated hdf5s."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Stack-Cube-Franka-IK-Rel-Mimic-Perturbed-v0",
        help="Gym id of the perturbed mimic env.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--results-csv",
        type=str,
        default="./docs/perturbation_sweep_v1_results.csv",
        help="Where to write the aggregated (std, n_success, n_failed, success_rate, wall_seconds) table.",
    )
    return parser.parse_args()


def sigma_tag(sigma: float) -> str:
    """Filesystem-safe tag for a sigma value, e.g. 0.1 -> 'std0p1'."""
    return f"std{str(sigma).replace('.', 'p').replace('-', 'neg')}"


def count_demos(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with h5py.File(path, "r") as f:
        return len(f["data"].keys()) if "data" in f else 0


def run_one_sigma(sigma: float, args) -> dict:
    tag = sigma_tag(sigma)
    output_file = os.path.join(args.output_dir, f"generated_{tag}.hdf5")
    failed_file = os.path.join(args.output_dir, f"generated_{tag}_failed.hdf5")

    env = os.environ.copy()
    env["PERTURB_STD"] = str(sigma)

    cmd = [
        "python",
        "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py",
        "--device",
        args.device,
        "--headless",
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--generation_num_trials",
        str(args.trials_per_sigma),
        "--input_file",
        args.input_file,
        "--output_file",
        output_file,
    ]

    print(f"\n=== sigma={sigma} (PERTURB_STD={sigma}) -> {output_file} ===", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall_seconds = time.time() - t0

    log_path = os.path.join(args.output_dir, f"log_{tag}.txt")
    with open(log_path, "w") as f:
        f.write(result.stdout)
        f.write("\n----- STDERR -----\n")
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"!!! sigma={sigma} FAILED (exit code {result.returncode}); see {log_path}", flush=True)

    n_success = count_demos(output_file)
    n_failed = count_demos(failed_file)
    total = n_success + n_failed
    success_rate = (n_success / total) if total > 0 else float("nan")

    print(
        f"sigma={sigma}: n_success={n_success} n_failed={n_failed} success_rate={success_rate:.3f}"
        f" wall_seconds={wall_seconds:.1f} exit_code={result.returncode}",
        flush=True,
    )

    return {
        "std": sigma,
        "n_success": n_success,
        "n_failed": n_failed,
        "success_rate": success_rate,
        "wall_seconds": round(wall_seconds, 1),
        "exit_code": result.returncode,
        "log_path": log_path,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)

    sigmas = [float(s) for s in args.sigmas.split(",")]
    rows = []
    for sigma in sigmas:
        rows.append(run_one_sigma(sigma, args))

    with open(args.results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["std", "n_success", "n_failed", "success_rate", "wall_seconds", "exit_code", "log_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote results table to {args.results_csv}")
    total_wall = sum(r["wall_seconds"] for r in rows)
    print(f"Total sweep wall time: {total_wall:.1f}s ({total_wall / 60:.1f} min)")


if __name__ == "__main__":
    main()
