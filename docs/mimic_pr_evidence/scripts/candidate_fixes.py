"""Compare two candidate fixes for the stack criterion firing on a cube in flight.

The criterion currently combines, per cube pair:
    xy_dist < xy_threshold
    h_dist - height_diff < height_threshold      <-- one-sided
    pos_diff[:, 2] < 0

With height_diff = 0.0468 (one cube) and height_threshold = 0.005, the surviving band on the
vertical gap is 0 < dz < 0.0518 -- anywhere from touching to 5.18 cm above. "Resting on top" and
"hovering above on the way down" are indistinguishable to it. The names (`height_threshold` next to
a `height_diff` that is exactly the cube size) read like a two-sided tolerance that lost its abs().

Candidates:
  A: two-sided height check, abs(h_dist - height_diff) < height_threshold
  B: require the cube to be at rest as well as in position

Judged on two populations, both of which must come out right:
  - accepted demos that end with the stack broken (should be rejected; these are the defect)
  - accepted demos that end with the stack intact  (must be kept; rejecting these breaks the yield)
"""

import h5py
import numpy as np

OPEN_VAL = 0.04
XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
DEFAULT_TOL = 1e-4 + 1e-4 * OPEN_VAL
DT = 0.05


def geom(cu, two_sided: bool):
    c1, c2, c3 = cu[:, 0:3], cu[:, 3:6], cu[:, 6:9]
    out = None
    for a, b in ((c1, c2), (c2, c3)):
        dd = a - b
        xy = np.linalg.norm(dd[:, :2], axis=1)
        h = np.linalg.norm(dd[:, 2:], axis=1)
        hcheck = (np.abs(h - H_DIFF) < H_TH) if two_sided else (h - H_DIFF < H_TH)
        ok = (xy < XY_TH) & hcheck & (dd[:, 2] < 0.0)
        out = ok if out is None else (out & ok)
    return out


def jaw_error(gp):
    return np.maximum(np.abs(gp[:, 0] - OPEN_VAL), np.abs(-gp[:, 1] - OPEN_VAL))


def speed(cu):
    """Magnitude of both moving cubes' velocity, finite-differenced from positions."""
    v2 = np.linalg.norm(np.gradient(cu[:, 3:6], DT, axis=0), axis=1)
    v3 = np.linalg.norm(np.gradient(cu[:, 6:9], DT, axis=0), axis=1)
    return np.maximum(v2, v3)


def evaluate(group):
    path = f"/home/pk/IsaacLab/datasets/pi05_training_data_v1/{group}/generated.hdf5"
    f = h5py.File(path, "r")
    d = f["data"]
    keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
    ends_broken, res = [], {}
    variants = {
        "current": lambda cu, gp: geom(cu, False) & (jaw_error(gp) <= DEFAULT_TOL),
        "A: two-sided height": lambda cu, gp: geom(cu, True) & (jaw_error(gp) <= DEFAULT_TOL),
        "B: at rest (<0.01 m/s)": lambda cu, gp: geom(cu, False) & (jaw_error(gp) <= DEFAULT_TOL) & (speed(cu) < 0.01),
        "A+B": lambda cu, gp: geom(cu, True) & (jaw_error(gp) <= DEFAULT_TOL) & (speed(cu) < 0.01),
    }
    for name in variants:
        res[name] = []
    for k in keys:
        cu = d[k]["obs"]["cube_positions"][:]
        gp = d[k]["obs"]["gripper_pos"][:]
        ends_broken.append(not bool(geom(cu, False)[-1]))
        for name, fn in variants.items():
            res[name].append(bool(fn(cu, gp).any()))
    f.close()

    broken = np.array(ends_broken)
    print(f"\n### {group} ###  accepted N={len(keys)}   of which end broken: {broken.sum()}")
    print(f"  {'variant':24s} {'keeps good':>12s} {'keeps broken':>14s}")
    for name in variants:
        v = np.array(res[name])
        good_kept = v[~broken].sum()
        bad_kept = v[broken].sum()
        print(
            f"  {name:24s} {good_kept:5d}/{(~broken).sum():<6d} {bad_kept:6d}/{broken.sum():<7d}"
            f"   -> yield {v.mean() * 100:5.1f}% of previously accepted"
        )


def main():
    for g in ("baseline", "arc_1p2cm"):
        evaluate(g)


if __name__ == "__main__":
    main()
