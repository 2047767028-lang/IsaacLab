# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sweep reset-time start-pose perturbation magnitude (PERTURB_STD) across seeds and measure
MimicGen generation success rate at each (sigma, seed) point, for the "contact-anchored
perturbation augmentation" validation experiment (docs/接触锚定扰动增广_设计记录.md).

Runs in **fixed-attempts** mode (PERTURB_FIXED_ATTEMPTS=1 on the child process): each
(sigma, seed) point runs exactly --attempts-per-point generation attempts, regardless of
success rate. This is required because DataGenConfig.generation_guarantee=True (the base
env's default) retries until N *successes* are banked with no bound on total attempts or
wall time - confirmed empirically (a sigma=5.0 probe took 337 attempts to bank 10 successes)
and in code (generation.py's env_loop only checks generation_guarantee/num_success/
num_attempts; DataGenConfig.max_num_failures is set by every mimic env cfg but read nowhere).

Resumable by design: each (sigma, seed) point's result is a row appended to that seed's CSV
immediately after the point finishes and is verified on disk, so a crash or restart only
loses the point in flight, not prior progress. Already-completed points (output+failed hdf5
both exist and their combined demo count equals --attempts-per-point) are skipped on rerun.

Usage:
    python scripts/imitation_learning/isaaclab_mimic/sweep_start_pose_perturbation.py \
        --sigmas 0.0,0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.65,0.80,1.00,1.30,1.60,2.00,2.50,3.00,4.00,5.50,8.00 \
        --seeds 1,2,3,4,5 \
        --attempts-per-point 500 \
        --num-envs 10
"""

import argparse
import csv
import os
import subprocess
import time

import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep start-pose perturbation magnitude x seed for MimicGen.")
    parser.add_argument("--sigmas", type=str, required=True, help="Comma-separated PERTURB_STD values.")
    parser.add_argument("--seeds", type=str, required=True, help="Comma-separated PERTURB_SEED values (ints).")
    parser.add_argument(
        "--attempts-per-point",
        type=int,
        required=True,
        help="Fixed number of generation attempts to run per (sigma, seed) point.",
    )
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--input-file", type=str, default="./datasets/annotated_dataset.hdf5")
    parser.add_argument("--output-dir", type=str, default="./datasets/perturbation_sweep_v1")
    parser.add_argument("--results-dir", type=str, default="./docs/perturbation_sweep_v1_results")
    parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-Mimic-Perturbed-v0")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def sigma_tag(sigma: float) -> str:
    return f"std{str(sigma).replace('.', 'p').replace('-', 'neg')}"


def count_demos(path: str) -> int | None:
    """Returns demo count, or None if the file is missing/unreadable (distinct from a real 0)."""
    if not os.path.exists(path):
        return None
    try:
        with h5py.File(path, "r") as f:
            return len(f["data"].keys()) if "data" in f else 0
    except OSError:
        # File exists but is not a valid/complete hdf5 (e.g. truncated by a crash mid-write).
        return None


def point_already_done(output_file: str, failed_file: str, expected_total: int) -> bool:
    ns = count_demos(output_file)
    nf = count_demos(failed_file)
    if ns is None or nf is None:
        return False
    return (ns + nf) == expected_total


def run_one_point(sigma: float, seed: int, args) -> dict:
    tag = sigma_tag(sigma)
    seed_dir = os.path.join(args.output_dir, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    output_file = os.path.join(seed_dir, f"generated_{tag}.hdf5")
    failed_file = os.path.join(seed_dir, f"generated_{tag}_failed.hdf5")
    log_path = os.path.join(seed_dir, f"log_{tag}.txt")

    if point_already_done(output_file, failed_file, args.attempts_per_point):
        ns = count_demos(output_file)
        nf = count_demos(failed_file)
        print(f"[skip, already done] seed={seed} sigma={sigma}: n_success={ns} n_failed={nf}", flush=True)
        return {
            "seed": seed,
            "std": sigma,
            "n_success": ns,
            "n_failed": nf,
            "n_attempts": ns + nf,
            "success_rate": ns / (ns + nf) if (ns + nf) > 0 else float("nan"),
            "wall_seconds": 0.0,
            "exit_code": 0,
            "verified_on_disk": True,
            "skipped": True,
        }

    env = os.environ.copy()
    env["PERTURB_STD"] = str(sigma)
    env["PERTURB_SEED"] = str(seed)
    env["PERTURB_FIXED_ATTEMPTS"] = "1"

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
        str(args.attempts_per_point),
        "--input_file",
        args.input_file,
        "--output_file",
        output_file,
    ]

    print(f"\n=== seed={seed} sigma={sigma} (PERTURB_STD={sigma} PERTURB_SEED={seed}) -> {output_file} ===", flush=True)
    t0 = time.time()
    # Redirect the child's stdout/stderr straight to a file instead of capture_output=True's
    # PIPEs. With PIPEs, the read end lives in this process; if this wrapper is ever killed
    # (e.g. by whatever mechanism killed the first sweep run, still unconfirmed), the child's
    # stdout/stderr pipes are closed out from under it, and if it does not handle that cleanly
    # (SIGPIPE/EPIPE on its next write) it can end up spinning instead of exiting. Writing
    # directly to a file has no reader to lose, so the child can't get stuck this way regardless
    # of what happens to this wrapper process.
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    wall_seconds = time.time() - t0

    if result.returncode != 0:
        print(f"!!! seed={seed} sigma={sigma} FAILED (exit code {result.returncode}); see {log_path}", flush=True)

    # Verify on disk - do not trust the exit code alone.
    ns = count_demos(output_file)
    nf = count_demos(failed_file)
    verified = ns is not None and nf is not None and (ns + nf) == args.attempts_per_point
    ns = ns or 0
    nf = nf or 0
    total = ns + nf
    success_rate = (ns / total) if total > 0 else float("nan")

    status = "OK" if verified else "MISMATCH/MISSING"
    print(
        f"seed={seed} sigma={sigma}: n_success={ns} n_failed={nf} n_attempts={total}"
        f" (expected {args.attempts_per_point}, disk_check={status})"
        f" success_rate={success_rate:.3f} wall_seconds={wall_seconds:.1f} exit_code={result.returncode}",
        flush=True,
    )

    return {
        "seed": seed,
        "std": sigma,
        "n_success": ns,
        "n_failed": nf,
        "n_attempts": total,
        "success_rate": success_rate,
        "wall_seconds": round(wall_seconds, 1),
        "exit_code": result.returncode,
        "verified_on_disk": verified,
        "skipped": False,
    }


FIELDNAMES = [
    "seed",
    "std",
    "n_success",
    "n_failed",
    "n_attempts",
    "success_rate",
    "wall_seconds",
    "exit_code",
    "verified_on_disk",
    "skipped",
]


def load_existing_rows(csv_path: str) -> dict:
    """Keyed by std, for a single seed's CSV. Used to skip re-appending duplicate rows on resume."""
    rows = {}
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["std"]] = row
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    sigmas = [float(s) for s in args.sigmas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    grand_total_wall = 0.0
    all_rows = []

    for seed in seeds:
        seed_csv = os.path.join(args.results_dir, f"seed{seed}.csv")
        already = load_existing_rows(seed_csv)
        need_header = not os.path.exists(seed_csv)

        with open(seed_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if need_header:
                writer.writeheader()

            for sigma in sigmas:
                if str(sigma) in already:
                    print(f"[skip, already in {seed_csv}] seed={seed} sigma={sigma}", flush=True)
                    row = already[str(sigma)]
                    all_rows.append(row)
                    continue

                row = run_one_point(sigma, seed, args)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                grand_total_wall += row["wall_seconds"]
                all_rows.append(row)

        # Post-seed verification pass: re-open every file for this seed from disk and confirm
        # counts, rather than trusting the in-loop bookkeeping alone.
        print(f"\n--- verifying seed={seed} on disk ---", flush=True)
        mismatches = []
        for sigma in sigmas:
            tag = sigma_tag(sigma)
            seed_dir = os.path.join(args.output_dir, f"seed{seed}")
            ns = count_demos(os.path.join(seed_dir, f"generated_{tag}.hdf5"))
            nf = count_demos(os.path.join(seed_dir, f"generated_{tag}_failed.hdf5"))
            if ns is None or nf is None or (ns + nf) != args.attempts_per_point:
                mismatches.append((sigma, ns, nf))
        if mismatches:
            print(f"!!! seed={seed} has {len(mismatches)} point(s) that failed disk verification: {mismatches}")
        else:
            print(f"seed={seed}: all {len(sigmas)} points verified on disk, counts match expected attempts.")

    combined_csv = os.path.join(args.results_dir, "combined.csv")
    with open(combined_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\nWrote combined results to {combined_csv}")
    print(f"Sum of measured wall_seconds this invocation: {grand_total_wall:.1f}s ({grand_total_wall / 3600:.2f}h)")
    print("(Skipped/already-done points contribute 0 to that sum; see per-seed CSVs for full history.)")


if __name__ == "__main__":
    main()
