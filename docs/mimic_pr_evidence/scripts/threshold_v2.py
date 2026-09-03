"""Choose the at-rest threshold from the signal the termination function actually reads.

Everything is replayed from the `states` group alone -- cube root poses, cube root velocities and
the robot joint positions that carry the gripper -- so there is no cross-group frame alignment to
get wrong. The replay is checked against the generator's own verdict first: if it does not mark
100% of the accepted demos as successes it is not faithful and nothing downstream of it is worth
reading.

Run D showed why this matters. The offline sweep used finite differences of 20 Hz positions, which
average over 50 ms; `root_lin_vel_w` is instantaneous, and a 0.01 m/s threshold on it cut the
generation success rate from 35.8% to about 11% -- far past the ~7-9% of demos that are defective.

Usage:  python threshold_v2.py <run.hdf5>
"""

import sys

import h5py
import numpy as np

XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
OPEN_VAL = 0.04
GRIP_TOL = 1e-4 + 1e-4 * OPEN_VAL
CUBES = ("cube_1", "cube_2", "cube_3")


def geom(p1, p2, p3):
    out = None
    for a, b in ((p1, p2), (p2, p3)):
        dd = a[:, :3] - b[:, :3]
        ok = np.linalg.norm(dd[:, :2], axis=1) < XY_TH
        ok &= (np.linalg.norm(dd[:, 2:3], axis=1) - H_DIFF < H_TH) & (dd[:, 2] < 0.0)
        out = ok if out is None else (out & ok)
    return out


def load(path):
    rows = []
    with h5py.File(path, "r") as f:
        d = f["data"]
        for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
            st = d[k]["states"]
            ro = st["rigid_object"]
            poses = [ro[c]["root_pose"][:] for c in CUBES]
            speed = np.stack([np.linalg.norm(ro[c]["root_velocity"][:, :3], axis=1) for c in CUBES]).max(axis=0)
            jp = st["articulation"]["robot"]["joint_position"][:]
            jaw = np.maximum(np.abs(jp[:, 7] - OPEN_VAL), np.abs(jp[:, 8] - OPEN_VAL))
            g = geom(*poses)
            rows.append({"k": k, "g": g, "q": g & (jaw <= GRIP_TOL), "v": speed})
    return rows


def main():
    path = sys.argv[1]
    rows = load(path)
    qual = np.array([r["q"].any() for r in rows])
    print(f"{path.split('/')[-1]}: demos={len(rows)}  replay reproduces the accepted verdict on {qual.sum()}")
    if not qual.all():
        print(f"  !!! {(~qual).sum()} do not reproduce -- replay is not faithful, stop here")
        return

    vmin = np.array([r["v"][r["q"]].min() for r in rows])
    broken = np.array([not bool(r["g"][-1]) for r in rows])
    print(f"  ends broken: {broken.sum()}/{len(rows)}\n")

    print("  cube speed on the most-settled qualifying frame (m/s):")
    for label, m in (("intact", ~broken), ("broken", broken)):
        if m.sum():
            q = np.percentile(vmin[m], [10, 50, 90])
            print(
                f"    {label:7s} n={m.sum():3d}  p10={q[0]:.4f}  p50={q[1]:.4f}"
                f"  p90={q[2]:.4f}  max={vmin[m].max():.4f}"
            )

    print("\n  threshold sweep (what the fix would keep):")
    print(f"    {'max_lin_vel':>12s} {'keeps intact':>14s} {'keeps broken':>14s} {'yield':>8s}")
    for t in (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0):
        p = vmin < t
        print(
            f"    {t:12.3f} {p[~broken].sum():6d}/{(~broken).sum():<7d}"
            f" {p[broken].sum():6d}/{broken.sum():<7d} {p.mean() * 100:7.1f}%"
        )


if __name__ == "__main__":
    main()
