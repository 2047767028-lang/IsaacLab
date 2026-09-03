"""Pick the fix by measurement rather than by which one sounds right.

Ground truth used here: an accepted demo whose stack is broken at the final frame is a defect (the
cube is on the table, a median 6.3 cm from where it should be); one whose stack is intact at the
final frame is a genuine demonstration that must survive the fix. A good fix rejects the former and
keeps the latter.

Variants:
  vel(t)   - the qualifying frame must also have both moving cubes below t m/s
  hold(k)  - the criterion must hold for k consecutive frames
  final    - the criterion must hold at the last frame of the episode
  abs      - two-sided height check

Velocity here is finite-differenced from the logged positions, which is noisier than the
root_lin_vel_w the real check would use; treat the exact threshold as indicative and confirm it in
simulation.
"""

import h5py
import numpy as np

OPEN_VAL = 0.04
XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
DEFAULT_TOL = 1e-4 + 1e-4 * OPEN_VAL
DT = 0.05


def geom(cu, two_sided=False):
    c1, c2, c3 = cu[:, 0:3], cu[:, 3:6], cu[:, 6:9]
    out = None
    for a, b in ((c1, c2), (c2, c3)):
        dd = a - b
        xy = np.linalg.norm(dd[:, :2], axis=1)
        h = np.linalg.norm(dd[:, 2:], axis=1)
        hc = (np.abs(h - H_DIFF) < H_TH) if two_sided else (h - H_DIFF < H_TH)
        ok = (xy < XY_TH) & hc & (dd[:, 2] < 0.0)
        out = ok if out is None else (out & ok)
    return out


def jaw_ok(gp):
    return np.maximum(np.abs(gp[:, 0] - OPEN_VAL), np.abs(-gp[:, 1] - OPEN_VAL)) <= DEFAULT_TOL


def speed(cu):
    v2 = np.linalg.norm(np.gradient(cu[:, 3:6], DT, axis=0), axis=1)
    v3 = np.linalg.norm(np.gradient(cu[:, 6:9], DT, axis=0), axis=1)
    return np.maximum(v2, v3)


def consecutive(mask, k):
    if k <= 1:
        return mask.any()
    run = 0
    for m in mask:
        run = run + 1 if m else 0
        if run >= k:
            return True
    return False


def load(group):
    path = f"/home/pk/IsaacLab/datasets/pi05_training_data_v1/{group}/generated.hdf5"
    f = h5py.File(path, "r")
    d = f["data"]
    eps = []
    for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
        cu = d[k]["obs"]["cube_positions"][:]
        gp = d[k]["obs"]["gripper_pos"][:]
        eps.append((cu, gp))
    f.close()
    return eps


def main():
    for group in ("baseline", "arc_1p2cm"):
        eps = load(group)
        broken = np.array([not bool(geom(cu)[-1]) for cu, _ in eps])
        base = np.array([bool((geom(cu) & jaw_ok(gp)).any()) for cu, gp in eps])
        assert base.all(), "every accepted demo must pass the current criterion"

        print(f"\n### {group} ###  N={len(eps)}  defects (end broken)={broken.sum()}")
        print(f"  {'variant':22s} {'good kept':>11s} {'defects kept':>13s} {'net yield':>10s}")

        def show(name, verdict):
            v = np.array(verdict)
            print(
                f"  {name:22s} {v[~broken].sum():4d}/{(~broken).sum():<6d}"
                f" {v[broken].sum():5d}/{broken.sum():<7d}"
                f" {v.mean() * 100:8.1f}%"
            )

        show("current", [True] * len(eps))
        show("abs (two-sided)", [bool((geom(cu, True) & jaw_ok(gp)).any()) for cu, gp in eps])
        for t in (0.005, 0.01, 0.02, 0.05, 0.10):
            show(f"vel < {t:.3f} m/s", [bool((geom(cu) & jaw_ok(gp) & (speed(cu) < t)).any()) for cu, gp in eps])
        for k in (3, 5, 10):
            show(f"hold {k} frames", [consecutive(geom(cu) & jaw_ok(gp), k) for cu, gp in eps])
        show("final frame", [bool((geom(cu) & jaw_ok(gp))[-1]) for cu, gp in eps])
        show(
            "final frame OR hold 5",
            [bool((geom(cu) & jaw_ok(gp))[-1]) or consecutive(geom(cu) & jaw_ok(gp), 5) for cu, gp in eps],
        )


if __name__ == "__main__":
    main()
