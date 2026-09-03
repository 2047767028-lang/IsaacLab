"""Offline reproduction of the Franka stack `cubes_stacked` success criterion.

Discipline note: this reproduction is validated against the *successes* first. A criterion
reproduction that has not been shown to pass 100% of known positives is not evidence about the
negatives.

Criterion (isaaclab_tasks .../stack/mdp/terminations.py::cubes_stacked):
  geometry: xy_dist(c1,c2) < 0.04, |h_dist(c1,c2)| - 0.0468 < 0.005, c1.z < c2.z, same for (c2,c3)
  gripper:  torch.isclose(finger_joint, gripper_open_val=0.04, atol=1e-4, rtol=1e-4) for BOTH jaws

The gripper_pos observation stores [finger_1, -finger_2] (see stack/mdp/observations.py), so the
second jaw has to be negated back before comparing to the open value.

Success is accumulated across the episode by the data generator
(`generated_success = generated_success or exec_success`), so "any frame satisfies it" is the
right semantics -- not "the final frame satisfies it".
"""

import sys

import h5py
import numpy as np

XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
OPEN_VAL = 0.04


def stacked_geom(cu: np.ndarray) -> np.ndarray:
    """cu: (T, 9) cube_positions. Returns (T,) bool for the geometry half of cubes_stacked.

    Env-origin offsets in cube_positions cancel out: every term is a difference between cubes.
    """
    c1, c2, c3 = cu[:, 0:3], cu[:, 3:6], cu[:, 6:9]
    d12, d23 = c1 - c2, c2 - c3
    xy12 = np.linalg.norm(d12[:, :2], axis=1)
    xy23 = np.linalg.norm(d23[:, :2], axis=1)
    h12 = np.linalg.norm(d12[:, 2:], axis=1)
    h23 = np.linalg.norm(d23[:, 2:], axis=1)
    ok = (xy12 < XY_TH) & (xy23 < XY_TH)
    ok &= (h12 - H_DIFF < H_TH) & (d12[:, 2] < 0.0)
    ok &= (h23 - H_DIFF < H_TH) & (d23[:, 2] < 0.0)
    return ok


def jaw_error(gp: np.ndarray) -> np.ndarray:
    """gp: (T, 2) gripper_pos obs = [finger_1, -finger_2]. Returns (T,) worst-jaw distance to open."""
    j0, j1 = gp[:, 0], -gp[:, 1]
    return np.maximum(np.abs(j0 - OPEN_VAL), np.abs(j1 - OPEN_VAL))


def scan(path: str, limit: int | None = None):
    f = h5py.File(path, "r")
    d = f["data"]
    keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
    if limit:
        keys = keys[:limit]
    geom_any = np.zeros(len(keys), dtype=bool)
    # Smallest jaw error over the frames where the geometry already holds. nan when geometry never
    # holds, which is a different failure mode entirely.
    best_gap = np.full(len(keys), np.nan)
    for i, k in enumerate(keys):
        cu = d[k]["obs"]["cube_positions"][:]
        gp = d[k]["obs"]["gripper_pos"][:]
        g = stacked_geom(cu)
        geom_any[i] = bool(g.any())
        if geom_any[i]:
            best_gap[i] = float(jaw_error(gp)[g].min())
    f.close()
    return keys, geom_any, best_gap


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    base = f"/home/pk/IsaacLab/datasets/pi05_training_data_v1/{group}"
    default_tol = 1e-4 + 1e-4 * OPEN_VAL  # torch.isclose: atol + rtol*|other|

    print(f"### {group} ###")
    print(f"criterion gripper tolerance = {default_tol * 1000:.4f} mm on a 40 mm jaw travel\n")

    keys, geom, gap = scan(f"{base}/generated.hdf5")
    passes = geom & (gap <= default_tol)
    print(f"[successes file] N={len(keys)}")
    print(f"  geometry satisfied at some frame : {geom.mean() * 100:6.2f}%")
    print(f"  full criterion (geometry+gripper): {passes.mean() * 100:6.2f}%   <- must be 100%")
    if passes.mean() < 1.0:
        bad = [keys[i] for i in np.where(~passes)[0]][:5]
        print(f"  !!! reproduction incomplete, first offenders: {bad}")
        return

    keys_f, geom_f, gap_f = scan(f"{base}/generated_failed.hdf5")
    n = len(keys_f)
    false_kill = geom_f & (gap_f > default_tol)
    print(f"\n[failures file] N={n}")
    print(f"  geometry never satisfied (real physical failure): {(~geom_f).sum():4d} ({(~geom_f).mean() * 100:5.2f}%)")
    print(f"  geometry satisfied but gripper check failed     : {false_kill.sum():4d} ({false_kill.mean() * 100:5.2f}%)")

    print("\n  how far the worst jaw actually stopped from open, among those:")
    g = gap_f[false_kill]
    for q in (50, 75, 90, 99):
        print(f"    p{q:<3d} = {np.percentile(g, q) * 1000:6.3f} mm")
    print(f"    max  = {g.max() * 1000:6.3f} mm")

    print("\n  recoverable share of the whole failure file, by relaxed gripper tolerance:")
    for tol_mm in (0.104, 0.5, 1.0, 2.0, 3.0, 5.0):
        rec = (geom_f & (gap_f <= tol_mm / 1000.0)).sum()
        print(
            f"    tol {tol_mm:5.3f} mm -> {rec:4d} recovered "
            f"({rec / n * 100:5.2f}% of failures, {rec / max(1, false_kill.sum()) * 100:5.1f}% of the false kills)"
        )


if __name__ == "__main__":
    main()
