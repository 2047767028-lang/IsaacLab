"""Do the accepted demos still have their cubes stacked when the episode ends?

`cubes_stacked` fires on an instantaneous configuration and the data generator accumulates it with
`generated_success = generated_success or exec_success`, so a single qualifying frame anywhere in
the episode marks the whole demo a success. A cube released above its target passes through the
qualifying configuration on the way down. If that is happening, some accepted demos should end with
the stack on the table -- and those demos are then written into the training dataset as
demonstrations of the task being completed.

Checks both delivered groups so the result cannot be an artifact of one generation run.
"""

import h5py
import numpy as np

OPEN_VAL = 0.04
XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
DEFAULT_TOL = 1e-4 + 1e-4 * OPEN_VAL
DT = 0.05


def stacked_geom(cu):
    c1, c2, c3 = cu[:, 0:3], cu[:, 3:6], cu[:, 6:9]
    d12, d23 = c1 - c2, c2 - c3
    ok = (np.linalg.norm(d12[:, :2], axis=1) < XY_TH) & (np.linalg.norm(d23[:, :2], axis=1) < XY_TH)
    ok &= (np.linalg.norm(d12[:, 2:], axis=1) - H_DIFF < H_TH) & (d12[:, 2] < 0.0)
    ok &= (np.linalg.norm(d23[:, 2:], axis=1) - H_DIFF < H_TH) & (d23[:, 2] < 0.0)
    return ok


def jaw_error(gp):
    return np.maximum(np.abs(gp[:, 0] - OPEN_VAL), np.abs(-gp[:, 1] - OPEN_VAL))


def run(group):
    path = f"/home/pk/IsaacLab/datasets/pi05_training_data_v1/{group}/generated.hdf5"
    f = h5py.File(path, "r")
    d = f["data"]
    keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
    rows = []
    for k in keys:
        cu = d[k]["obs"]["cube_positions"][:]
        gp = d[k]["obs"]["gripper_pos"][:]
        g = stacked_geom(cu)
        err = jaw_error(gp)
        q = g & (err <= DEFAULT_TOL)
        v = np.gradient(cu[:, 8], DT)
        i = np.where(q)[0]
        rows.append(
            {
                "k": k,
                "holds_at_end": bool(g[-1]),
                "vz_best": float(np.min(np.abs(v[i]))) if len(i) else np.nan,
                "xy23_end": float(np.linalg.norm((cu[-1, 3:6] - cu[-1, 6:9])[:2])),
                "dz23_end": float(cu[-1, 5] - cu[-1, 8]),
            }
        )
    f.close()

    n = len(rows)
    end_ok = np.array([r["holds_at_end"] for r in rows])
    vz = np.array([r["vz_best"] for r in rows])
    print(f"\n### {group} ### accepted demos N={n}")
    print(f"  stack still holds at the final frame : {end_ok.sum():3d} ({end_ok.mean() * 100:5.2f}%)")
    print(f"  stack broken by the final frame      : {(~end_ok).sum():3d} ({(~end_ok).mean() * 100:5.2f}%)")

    bad = ~end_ok
    if bad.any():
        print("\n  the ones that end broken:")
        print(f"    cube_3 speed at the qualifying frame: median {np.median(vz[bad]):.4f} m/s "
              f"(vs {np.median(vz[~bad]):.4f} for the rest)")
        xy = np.array([r["xy23_end"] for r in rows])[bad]
        dz = np.array([r["dz23_end"] for r in rows])[bad]
        print(f"    final xy distance cube_2->cube_3  : median {np.median(xy) * 100:.2f} cm (threshold 4 cm)")
        print(f"    final height difference           : median {np.median(dz) * 100:.2f} cm (a stack reads -4.68)")
        names = [rows[i]["k"] for i in np.where(bad)[0]][:10]
        print(f"    demos: {names}")
    return end_ok, vz


def main():
    for group in ("baseline", "arc_1p2cm"):
        run(group)


if __name__ == "__main__":
    main()
