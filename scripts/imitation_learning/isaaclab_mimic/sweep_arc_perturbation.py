# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sweep arc-perturbation magnitude (PERTURB_ARC_STD, applied to every subtask's target
trajectory - see `_apply_arc_perturbation` in isaaclab_mimic/datagen/data_generator.py) across
seeds and measure MimicGen generation success rate at each (arc_std, seed) point. This is the
"主线二·扩展到全部subtask轨迹多样化" v2 experiment (see CLAUDE.md), separate from and not
replacing `sweep_start_pose_perturbation.py` (the v1/start-pose-only experiment - that script and
its results are left untouched).

Magnitudes are given on the CLI in **centimeters** (matching how the small-scale probe results
were reported/discussed) and converted to meters for PERTURB_ARC_STD, which is what
`_apply_arc_perturbation` actually expects. `PERTURB_STD` (the old reset-time joint-noise
mechanism from the v1 experiment) is deliberately forced to 0 on every child process, to isolate
the arc mechanism - the two are independent and additive, but this sweep only wants the arc
effect. No new env cfg/gym id is needed: PERTURB_ARC_STD/PERTURB_ARC_FREEZE_FRAC are read
directly by data_generator.py at generation time, for whatever task is passed.

Same verified patterns as sweep_start_pose_perturbation.py, reused deliberately rather than
reinvented:
- fixed-attempts mode (PERTURB_FIXED_ATTEMPTS=1): generation_guarantee=True (the default) retries
  until N successes are banked with no bound on total attempts/wall time at low success rates -
  confirmed both empirically and in code during the v1 experiment. Fixed-attempts makes every
  point's cost independent of its success rate.
- child stdout/stderr redirected straight to a log file, never subprocess PIPEs
  (capture_output=True) - a wrapper killed while its child holds a PIPE whose read end lives in
  the (now-dead) wrapper can leave the child spinning on SIGPIPE/EPIPE instead of exiting; a file
  has no reader to lose. (Root-caused during the v1 experiment; see CLAUDE.md and commit
  cac8766d9.)
- resumable by design: each (arc_std, seed) point's result is appended+fsync'd to that seed's CSV
  immediately after the point finishes and is independently verified against the actual hdf5 demo
  counts on disk (not the exit code), so a crash only loses the point in flight. Already-complete
  points are skipped on rerun.
- long runs should be launched via `setsid nohup ... </dev/null & disown`, NOT this process's own
  `run_in_background`-equivalent mechanisms - a long run_in_background-tracked task was killed by
  an unconfirmed mechanism after ~61 minutes during the v1 experiment; setsid fully detaches the
  process tree from that tracking.

Usage:
    python scripts/imitation_learning/isaaclab_mimic/sweep_arc_perturbation.py \
        --arc-stds-cm 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.8,2.0,2.5,3.0 \
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
    parser = argparse.ArgumentParser(description="Sweep arc-perturbation magnitude x seed for MimicGen.")
    parser.add_argument(
        "--arc-stds-cm", type=str, required=True, help="Comma-separated PERTURB_ARC_STD values, in centimeters."
    )
    parser.add_argument("--seeds", type=str, required=True, help="Comma-separated PERTURB_SEED values (ints).")
    parser.add_argument(
        "--attempts-per-point",
        type=int,
        required=True,
        help="Fixed number of generation attempts to run per (arc_std, seed) point.",
    )
    parser.add_argument(
        "--freeze-frac", type=float, default=0.3, help="PERTURB_ARC_FREEZE_FRAC, fixed across this sweep."
    )
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--input-file", type=str, default="./datasets/annotated_dataset.hdf5")
    parser.add_argument("--output-dir", type=str, default="./datasets/perturbation_sweep_v2_arc")
    parser.add_argument("--results-dir", type=str, default="./docs/perturbation_sweep_v2_arc_results")
    parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-Mimic-Perturbed-v0")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def cm_tag(std_cm: float) -> str:
    return f"cm{str(std_cm).replace('.', 'p').replace('-', 'neg')}"


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


def run_one_point(std_cm: float, seed: int, args) -> dict:
    tag = cm_tag(std_cm)
    seed_dir = os.path.join(args.output_dir, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    output_file = os.path.join(seed_dir, f"generated_{tag}.hdf5")
    failed_file = os.path.join(seed_dir, f"generated_{tag}_failed.hdf5")
    log_path = os.path.join(seed_dir, f"log_{tag}.txt")

    if point_already_done(output_file, failed_file, args.attempts_per_point):
        ns = count_demos(output_file)
        nf = count_demos(failed_file)
        print(f"[skip, already done] seed={seed} arc_std={std_cm}cm: n_success={ns} n_failed={nf}", flush=True)
        return {
            "seed": seed,
            "arc_std_cm": std_cm,
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
    env["PERTURB_STD"] = "0"  # isolate the arc mechanism from the v1 start-pose joint-noise one
    env["PERTURB_SEED"] = str(seed)
    env["PERTURB_FIXED_ATTEMPTS"] = "1"
    env["PERTURB_ARC_STD"] = str(std_cm / 100.0)  # cm -> meters, what _apply_arc_perturbation expects
    env["PERTURB_ARC_FREEZE_FRAC"] = str(args.freeze_frac)

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

    print(
        f"\n=== seed={seed} arc_std={std_cm}cm ({env['PERTURB_ARC_STD']}m, "
        f"freeze_frac={args.freeze_frac}) -> {output_file} ===",
        flush=True,
    )
    t0 = time.time()
    # Direct-to-file redirect, not capture_output=True/PIPEs - see module docstring.
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    wall_seconds = time.time() - t0

    if result.returncode != 0:
        print(f"!!! seed={seed} arc_std={std_cm}cm FAILED (exit code {result.returncode}); see {log_path}", flush=True)

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
        f"seed={seed} arc_std={std_cm}cm: n_success={ns} n_failed={nf} n_attempts={total}"
        f" (expected {args.attempts_per_point}, disk_check={status})"
        f" success_rate={success_rate:.3f} wall_seconds={wall_seconds:.1f} exit_code={result.returncode}",
        flush=True,
    )

    return {
        "seed": seed,
        "arc_std_cm": std_cm,
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
    "arc_std_cm",
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
    """Keyed by arc_std_cm, for a single seed's CSV. Used to skip re-appending duplicate rows on resume."""
    rows = {}
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["arc_std_cm"]] = row
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    arc_stds_cm = [float(s) for s in args.arc_stds_cm.split(",")]
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

            for std_cm in arc_stds_cm:
                if str(std_cm) in already:
                    print(f"[skip, already in {seed_csv}] seed={seed} arc_std={std_cm}cm", flush=True)
                    row = already[str(std_cm)]
                    all_rows.append(row)
                    continue

                row = run_one_point(std_cm, seed, args)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                grand_total_wall += row["wall_seconds"]
                all_rows.append(row)

        # Post-seed verification pass: re-open every file for this seed from disk and confirm
        # counts, rather than trusting the in-loop bookkeeping alone.
        print(f"\n--- verifying seed={seed} on disk ---", flush=True)
        mismatches = []
        for std_cm in arc_stds_cm:
            tag = cm_tag(std_cm)
            seed_dir = os.path.join(args.output_dir, f"seed{seed}")
            ns = count_demos(os.path.join(seed_dir, f"generated_{tag}.hdf5"))
            nf = count_demos(os.path.join(seed_dir, f"generated_{tag}_failed.hdf5"))
            if ns is None or nf is None or (ns + nf) != args.attempts_per_point:
                mismatches.append((std_cm, ns, nf))
        if mismatches:
            print(f"!!! seed={seed} has {len(mismatches)} point(s) that failed disk verification: {mismatches}")
        else:
            print(f"seed={seed}: all {len(arc_stds_cm)} points verified on disk, counts match expected attempts.")

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
