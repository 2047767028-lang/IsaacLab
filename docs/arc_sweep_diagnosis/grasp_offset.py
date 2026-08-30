"""If the arm converges at placement time and placement still fails, what is carrying the error?

The noise-free end-of-subtask dwell got the arm to within 0.46 cm of its target (p05, against 2.50
for the reference) and placement failures went up rather than down, 154 to 166. So the arm's
position when it lets go is not what decides the placement.

The obvious remaining carrier is the cube itself: if the perturbed approach makes the gripper close
off-centre, the cube sits crooked in the jaws and is placed crooked no matter where the arm is. That
offset is measurable -- the cube's position relative to the gripper, right after the jaws close.
"""

import h5py
import numpy as np

OUT = "/home/pk/.claude/jobs/10fee75c/tmp/out"
OPEN_VAL = 0.04
XY, HT, HD = 0.04, 0.005, 0.0468


def grasp_frame(grip):
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = np.where(jaw < OPEN_VAL - 0.001)[0]
    return int(closed[0]) if len(closed) else None


def scan(tag):
    rows = []
    for suffix, ok in (("", True), ("_failed", False)):
        try:
            f = h5py.File(f"{OUT}/fix_{tag}{suffix}.hdf5", "r")
        except OSError:
            continue
        with f:
            d = f["data"]
            for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
                obs, ro = d[k]["obs"], d[k]["states"]["rigid_object"]
                g = grasp_frame(obs["gripper_pos"][:])
                if g is None:
                    continue
                eef = obs["eef_pos"][:]
                c1 = ro["cube_1"]["root_pose"][:, :3]
                c2 = ro["cube_2"]["root_pose"][:, :3]
                # a few frames after the jaws close, once the grasp has settled
                i = min(g + 5, len(eef) - 1)
                held = c2[i] - eef[i]
                if np.linalg.norm(held) > 0.15:  # jaws closed on nothing
                    continue
                dd = c1[-1] - c2[-1]
                placed = (
                    np.linalg.norm(dd[:2]) < XY and abs(dd[2]) - HD < HT and dd[2] < 0
                )
                rows.append(
                    {
                        "off": float(np.linalg.norm(held[:2])),  # lateral offset in the jaws
                        "placed": bool(placed),
                        "err": float(np.linalg.norm(dd[:2])),
                    }
                )
    return rows


def main():
    print("cube_2's lateral offset inside the jaws, measured 5 frames after they close\n")
    print(f"  {'run':<14s} {'n':>5s} {'offset (placed)':>17s} {'offset (misplaced)':>20s} {'d':>7s} {'corr with err':>15s}")
    for tag in ("ref_low", "ref_high", "tail2_high", "gate1_high"):
        rows = scan(tag)
        if not rows:
            print(f"  {tag:<14s} (no data)")
            continue
        off = np.array([r["off"] for r in rows])
        ok = np.array([r["placed"] for r in rows])
        err = np.array([r["err"] for r in rows])
        a, b = off[ok], off[~ok]
        d = (b.mean() - a.mean()) / (np.sqrt((a.var() + b.var()) / 2) or 1e-12) if len(b) else float("nan")
        corr = float(np.corrcoef(off, err)[0, 1])
        print(
            f"  {tag:<14s} {len(rows):5d} {np.median(a) * 100:16.3f}cm"
            f" {(np.median(b) * 100 if len(b) else float('nan')):19.3f}cm {d:7.2f} {corr:15.3f}"
        )

    print("\n  a large separation, or a correlation with the final placement error, would mean the")
    print("  cube's seat in the jaws is what carries the damage forward.")


if __name__ == "__main__":
    main()
