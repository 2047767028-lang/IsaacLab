"""Does restoring the joint configuration buy anything that restoring the gripper pose does not?

Restoring the joints implies restoring the gripper pose -- forward kinematics gives one from the
other -- so direction 1 is sufficient by construction. The question is whether it is *necessary*:
whether the joint configuration carries any information about the grasp beyond what the gripper's
own position already carries. If it does not, direction 1 costs a parallel generation run and a
joint-space blend to buy exactly what directions 2 and 3 buy far more cheaply.

Stratifying is the honest way to ask this. Inside a narrow band of gripper-position error the
gripper pose is held roughly constant, so any remaining separation by joint error is joint error's
own contribution.
"""

import sys

import h5py
import numpy as np

ROOT = "/home/pk/IsaacLab/datasets/perturbation_sweep_v2_arc"
CUBES = ("cube_1", "cube_2", "cube_3")
OPEN_VAL = 0.04


def key_of(ro):
    return tuple(np.round(np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]), 6))


def grasp_frame(g):
    jaw = np.minimum(g[:, 0], -g[:, 1])
    closed = np.where(jaw < OPEN_VAL - 0.001)[0]
    return int(closed[0]) if len(closed) else None


def load_run(seed, cm):
    tag = f"cm{cm:.1f}".replace(".", "p")
    out = {}
    for suffix, ok in (("", True), ("_failed", False)):
        with h5py.File(f"{ROOT}/seed{seed}/generated_{tag}{suffix}.hdf5", "r") as f:
            d = f["data"]
            for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
                obs, ro = d[k]["obs"], d[k]["states"]["rigid_object"]
                out[key_of(ro)] = {
                    "success": ok,
                    "pos": obs["eef_pos"][:],
                    "joints": obs["joint_pos"][:, :7],
                    "grip": obs["gripper_pos"][:],
                    "cube2": ro["cube_2"]["root_pose"][:, :3],
                }
    return out


def collect(seed, cms):
    ref = load_run(seed, 0.5)
    rows = []
    for cm in cms:
        arc = load_run(seed, cm)
        for k, a in arc.items():
            r = ref.get(k)
            if r is None:
                continue
            ga, gr = grasp_frame(a["grip"]), grasp_frame(r["grip"])
            if ga is None or gr is None:
                continue
            rows.append(
                {
                    "picked": float(np.linalg.norm(a["cube2"][-1] - a["cube2"][0])) > 0.02,
                    "dpos": float(np.linalg.norm(a["pos"][ga] - r["pos"][gr])),
                    "djoint": float(np.linalg.norm(a["joints"][ga] - r["joints"][gr])),
                }
            )
    return rows


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cms = [1.5, 2.0, 2.5, 3.0]  # pooled, to have enough failed grasps to stratify
    rows = collect(seed, cms)
    picked = np.array([r["picked"] for r in rows])
    dpos = np.array([r["dpos"] for r in rows])
    djoint = np.array([r["djoint"] for r in rows])
    print(f"seed={seed}  amplitudes {cms} pooled  n={len(rows)}  grasp failures={int((~picked).sum())}\n")

    print("=== unstratified: both quantities separate the outcome ===")
    for name, v, unit, s in (("gripper position", dpos, "cm", 100), ("joint configuration", djoint, "rad", 1)):
        a, b = v[picked], v[~picked]
        d = (b.mean() - a.mean()) / (np.sqrt((a.var() + b.var()) / 2) or 1e-12)
        print(f"  {name:<20s} picked {np.median(a) * s:7.3f}{unit}   failed {np.median(b) * s:7.3f}{unit}   d={d:.2f}")

    print("\n=== stratified by gripper-position error: does joint error still say anything? ===")
    edges = np.quantile(dpos, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"  {'gripper-error band':<24s} {'n':>5s} {'fail':>5s} {'joint (picked)':>15s} {'joint (failed)':>15s} {'d':>7s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dpos >= lo) & (dpos < hi if hi != edges[-1] else dpos <= hi)
        if m.sum() < 20 or (~picked[m]).sum() < 3:
            continue
        a, b = djoint[m & picked], djoint[m & ~picked]
        d = (b.mean() - a.mean()) / (np.sqrt((a.var() + b.var()) / 2) or 1e-12)
        print(
            f"  {lo * 100:6.2f}-{hi * 100:6.2f} cm{'':<8s} {m.sum():5d} {(~picked[m]).sum():5d}"
            f" {np.median(a):14.4f}  {np.median(b):14.4f} {d:7.2f}"
        )

    print("\n  If the per-band d values collapse toward 0 while the unstratified d was large, the")
    print("  joint configuration was only ever a proxy for the gripper position, and restoring it")
    print("  buys nothing extra.")


if __name__ == "__main__":
    main()
