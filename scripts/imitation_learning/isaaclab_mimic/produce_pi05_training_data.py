# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Produce two fixed-config datasets (baseline, arc-perturbed) with camera observations for
pi0.5 training on the user's lab server - CLAUDE.md "主线二·第二层验证启动". This is a
production data run, not a diagnostic sweep: two groups, each a single fixed-attempts
generation call, not a (magnitude x seed) grid - so it deliberately does not reuse
sweep_arc_perturbation.py's grid structure, only the safety patterns validated there:

- fixed-attempts mode (PERTURB_FIXED_ATTEMPTS=1): generation_guarantee=True (default) retries
  until N successes are banked with no bound on attempts/wall time at low success rates.
- child stdout/stderr redirected straight to a log file, never subprocess PIPEs
  (capture_output=True) - a wrapper killed while its child holds a PIPE whose read end lives in
  the wrapper can leave the child spinning on SIGPIPE/EPIPE instead of exiting.
- resumable: a group already on disk with n_success+n_failed == its attempt budget is skipped,
  verified by reopening the hdf5 files and recounting (never trusting an exit code alone).
- long runs launched via `setsid nohup ... </dev/null & disown`, not this process's own
  backgrounding - an earlier long run tracked by the harness's own mechanism was killed by an
  unconfirmed cause after ~61 minutes.

New for this run (production data, not diagnostics): after a group finishes, spot-checks that a
sample of successful demos' camera frames are not degenerate (all-zero / all-one-value), since
this data is going to actually be trained on, not just counted.

Usage:
    python scripts/imitation_learning/isaaclab_mimic/produce_pi05_training_data.py
(no CLI args - the two groups and their configs are the point of this script, not something to
parameterize away; see GROUPS below.)
"""

import os
import subprocess
import time

import h5py
import numpy as np

TASK = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-Perturbed-v0"
INPUT_FILE = "./datasets/annotated_dataset.hdf5"
OUTPUT_ROOT = "./datasets/pi05_training_data_v1"
NUM_ENVS = 10
ATTEMPTS_BUDGET = 1050
CAMERA_KEYS = ["table_cam", "wrist_cam"]

GROUPS = {
    "baseline": {
        "PERTURB_STD": "0",
        "PERTURB_ARC_STD": "0",
    },
    "arc_1p2cm": {
        "PERTURB_STD": "0",
        "PERTURB_ARC_STD": "0.012",
        "PERTURB_ARC_FREEZE_FRAC": "0.3",
    },
}


def count_demos(path: str) -> int | None:
    if not os.path.exists(path):
        return None
    try:
        with h5py.File(path, "r") as f:
            return len(f["data"].keys()) if "data" in f else 0
    except OSError:
        return None


def group_already_done(output_file: str, failed_file: str, expected_total: int) -> bool:
    ns = count_demos(output_file)
    nf = count_demos(failed_file)
    if ns is None or nf is None:
        return False
    return (ns + nf) == expected_total


def check_image_integrity(output_file: str, sample_n: int = 5) -> dict:
    """Reopen a sample of successful demos and check each camera key's first frame is not
    degenerate (all-zero or all-one-value), per demo, per camera. Returns a report dict;
    does not raise - a caller decides what a failure means."""
    report = {"checked_demos": 0, "issues": []}
    with h5py.File(output_file, "r") as f:
        if "data" not in f:
            report["issues"].append("no 'data' group in file")
            return report
        demo_keys = list(f["data"].keys())[:sample_n]
        for dk in demo_keys:
            demo = f["data"][dk]
            for cam in CAMERA_KEYS:
                path = f"obs/{cam}"
                if path not in demo:
                    report["issues"].append(f"{dk}: missing obs/{cam}")
                    continue
                frame = np.asarray(demo[path][0])
                mean, std, mn, mx = float(frame.mean()), float(frame.std()), int(frame.min()), int(frame.max())
                degenerate = std < 1e-6 or (mn == mx)
                report["checked_demos"] += 1
                if degenerate:
                    report["issues"].append(
                        f"{dk}/{cam}: DEGENERATE frame - mean={mean:.2f} std={std:.4f} min={mn} max={mx}"
                    )
    return report


def run_group(name: str, env_overrides: dict) -> dict:
    group_dir = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(group_dir, exist_ok=True)
    output_file = os.path.join(group_dir, "generated.hdf5")
    failed_file = os.path.join(group_dir, "generated_failed.hdf5")
    log_path = os.path.join(group_dir, "log.txt")
    result_path = os.path.join(group_dir, "result.txt")

    if group_already_done(output_file, failed_file, ATTEMPTS_BUDGET):
        ns, nf = count_demos(output_file), count_demos(failed_file)
        print(f"[skip, already done] group={name}: n_success={ns} n_failed={nf}", flush=True)
        return {"group": name, "n_success": ns, "n_failed": nf, "wall_seconds": 0.0, "skipped": True}

    env = os.environ.copy()
    env["PERTURB_SEED"] = "1"
    env["PERTURB_FIXED_ATTEMPTS"] = "1"
    env.update(env_overrides)

    cmd = [
        "python",
        "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py",
        "--device",
        "cuda",
        "--enable_cameras",
        "--headless",
        "--task",
        TASK,
        "--num_envs",
        str(NUM_ENVS),
        "--generation_num_trials",
        str(ATTEMPTS_BUDGET),
        "--input_file",
        INPUT_FILE,
        "--output_file",
        output_file,
    ]

    print(f"\n=== group={name} env_overrides={env_overrides} -> {output_file} ===", flush=True)
    t0 = time.time()
    # Direct-to-file redirect, not capture_output=True/PIPEs - see module docstring.
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    wall_seconds = time.time() - t0

    if result.returncode != 0:
        print(f"!!! group={name} FAILED (exit code {result.returncode}); see {log_path}", flush=True)

    ns = count_demos(output_file)
    nf = count_demos(failed_file)
    verified = ns is not None and nf is not None and (ns + nf) == ATTEMPTS_BUDGET
    ns, nf = ns or 0, nf or 0
    total = ns + nf

    print(
        f"group={name}: n_success={ns} n_failed={nf} n_attempts={total} (expected {ATTEMPTS_BUDGET},"
        f" disk_check={'OK' if verified else 'MISMATCH/MISSING'}) wall_seconds={wall_seconds:.1f}"
        f" exit_code={result.returncode}",
        flush=True,
    )

    img_report = {"checked_demos": 0, "issues": ["skipped: 0 successful demos"]}
    if ns > 0:
        img_report = check_image_integrity(output_file)
        if img_report["issues"]:
            print(f"!!! group={name} image integrity issues: {img_report['issues']}", flush=True)
        else:
            print(f"group={name}: image integrity OK on {img_report['checked_demos']} sampled demo/camera frames.")

    disk_bytes = 0
    for fn in [output_file, failed_file]:
        if os.path.exists(fn):
            disk_bytes += os.path.getsize(fn)

    row = {
        "group": name,
        "n_success": ns,
        "n_failed": nf,
        "n_attempts": total,
        "wall_seconds": round(wall_seconds, 1),
        "exit_code": result.returncode,
        "verified_on_disk": verified,
        "disk_bytes": disk_bytes,
        "image_check_issues": img_report["issues"],
        "image_check_n": img_report["checked_demos"],
        "skipped": False,
    }
    with open(result_path, "w") as f:
        for k, v in row.items():
            f.write(f"{k}: {v}\n")
    return row


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_results = []
    for name, overrides in GROUPS.items():
        all_results.append(run_group(name, overrides))

    print("\n=== summary ===")
    for r in all_results:
        print(r)


if __name__ == "__main__":
    main()
