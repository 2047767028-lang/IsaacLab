"""Pick the at-rest threshold from the velocity the fix actually reads.

The offline sweep used finite differences of the 20 Hz logged positions, which average over 50 ms.
The termination function reads `root_lin_vel_w`, an instantaneous value at the env step, and run D
showed the difference matters: at 0.01 m/s the generation success rate fell from 35.8% to ~11%,
far more than the ~7% of demos that are actually defective.

The recorded `states/rigid_object/<cube>/root_velocity` is that same signal, logged per frame, so
the threshold can be chosen from a run that used the stock criterion rather than guessed.

Ground truth per demo: the stack is either standing at the final frame or it is not. A useful
threshold rejects the demos where it is not and keeps the ones where it is.

Usage:  python threshold_from_states.py <run.hdf5>
"""

import sys

import h5py
import numpy as np

XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
OPEN_VAL = 0.04
GRIP_TOL = 1e-4 + 1e-4 * OPEN_VAL


def geom_from_states(pose1, pose2, pose3):
    """Replay the positional half of cubes_stacked from the recorded root poses."""
    out = None
    for a, b in ((pose1, pose2), (pose2, pose3)):
        dd = a[:, :3] - b[:, :3]
        ok = np.linalg.norm(dd[:, :2], axis=1) < XY_TH
        ok &= (np.linalg.norm(dd[:, 2:3], axis=1) - H_DIFF < H_TH) & (dd[:, 2] < 0.0)
        out = ok if out is None else (out & ok)
    return out


def main():
    path = sys.argv[1]
    with h5py.File(path, "r") as f:
        d = f["data"]
        keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
        rows = []
        for k in keys:
            st = d[k]["states"]["rigid_object"]
            p1 = st["cube_1"]["root_pose"][:]
            p2 = st["cube_2"]["root_pose"][:]
            p3 = st["cube_3"]["root_pose"][:]
            v = np.stack(
                [np.linalg.norm(st[c]["root_velocity"][:, :3], axis=1) for c in ("cube_1", "cube_2", "cube_3")]
            ).max(axis=0)
            gp = d[k]["obs"]["gripper_pos"][:]
            jaw = np.maximum(np.abs(gp[:, 0] - OPEN_VAL), np.abs(-gp[:, 1] - OPEN_VAL))
            g = geom_from_states(p1, p2, p3)
            q = g & (jaw <= GRIP_TOL)
            rows.append(
                {
                    "k": k,
                    "qualifies": bool(q.any()),
                    # slowest the cubes ever were on a qualifying frame -- what a threshold sees
                    "vmin": float(v[q].min()) if q.any() else np.nan,
                    "broken": not bool(g[-1]),
                }
            )

    qual = np.array([r["qualifies"] for r in rows])
    print(f"{path.split('/')[-1]}: demos={len(rows)}  reproduce the stock criterion on {qual.sum()}")
    if not qual.all():
        print(f"  !!! {(~qual).sum()} accepted demos do not reproduce -- replay is not faithful, stop here")
        return

    vmin = np.array([r["vmin"] for r in rows])
    broken = np.array([r["broken"] for r in rows])
    print(f"  ends broken: {broken.sum()}/{len(rows)}\n")

    print("  cube speed on the most-settled qualifying frame (m/s):")
    for label, m in (("intact", ~broken), ("broken", broken)):
        if m.sum():
            q = np.percentile(vmin[m], [10, 50, 90])
            print(f"    {label:7s} n={m.sum():3d}  p10={q[0]:.4f}  p50={q[1]:.4f}  p90={q[2]:.4f}  max={vmin[m].max():.4f}")

    print("\n  threshold sweep:")
    print(f"    {'max_lin_vel':>12s} {'keeps intact':>13s} {'keeps broken':>13s} {'yield':>8s}")
    for t in (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        passes = vmin < t
        print(
            f"    {t:12.3f} {passes[~broken].sum():5d}/{(~broken).sum():<7d}"
            f" {passes[broken].sum():5d}/{broken.sum():<7d} {passes.mean() * 100:7.1f}%"
        )


if __name__ == "__main__":
    main()
