"""What state do the gripper-rejected episodes actually end in?

The timing diagnostic showed geometry never holds at the final frame in any of the 76, while the
jaws do reach fully open. So the question the criterion is really answering is not "was the tolerance
too tight" but "did the stack survive the release". This quantifies the end state so the two can be
told apart on evidence rather than on the size of a tolerance.

Reference scales: cube side 4.68 cm; the xy threshold is 4 cm; a cube resting on the table rather
than on its neighbour shows up as a height difference near zero instead of near 4.68 cm.
"""

import h5py
import numpy as np

OPEN_VAL = 0.04
XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
DEFAULT_TOL = 1e-4 + 1e-4 * OPEN_VAL


def terms(cu):
    c1, c2, c3 = cu[:, 0:3], cu[:, 3:6], cu[:, 6:9]
    d12, d23 = c1 - c2, c2 - c3
    return d12, d23


def stacked_geom(cu):
    d12, d23 = terms(cu)
    ok = (np.linalg.norm(d12[:, :2], axis=1) < XY_TH) & (np.linalg.norm(d23[:, :2], axis=1) < XY_TH)
    ok &= (np.linalg.norm(d12[:, 2:], axis=1) - H_DIFF < H_TH) & (d12[:, 2] < 0.0)
    ok &= (np.linalg.norm(d23[:, 2:], axis=1) - H_DIFF < H_TH) & (d23[:, 2] < 0.0)
    return ok


def jaw_error(gp):
    return np.maximum(np.abs(gp[:, 0] - OPEN_VAL), np.abs(-gp[:, 1] - OPEN_VAL))


def main():
    path = "/home/pk/IsaacLab/datasets/pi05_training_data_v1/baseline/generated_failed.hdf5"
    f = h5py.File(path, "r")
    d = f["data"]
    rows = []
    for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
        cu = d[k]["obs"]["cube_positions"][:]
        gp = d[k]["obs"]["gripper_pos"][:]
        g = stacked_geom(cu)
        if not g.any():
            continue
        err = jaw_error(gp)
        best = err[g].min()
        if best <= DEFAULT_TOL:
            continue
        d12, d23 = terms(cu)
        rows.append(
            {
                "k": k,
                "best": best,
                # end state
                "xy12": float(np.linalg.norm(d12[-1, :2])),
                "xy23": float(np.linalg.norm(d23[-1, :2])),
                "z12": float(d12[-1, 2]),
                "z23": float(d23[-1, 2]),
                # how much the top cube moved between the last geometry frame and the end
                "drift3": float(np.linalg.norm(cu[-1, 6:9] - cu[np.where(g)[0][-1], 6:9])),
                "drift2": float(np.linalg.norm(cu[-1, 3:6] - cu[np.where(g)[0][-1], 3:6])),
            }
        )
    f.close()

    n = len(rows)
    best = np.array([r["best"] for r in rows])
    print(f"gripper-rejected episodes: N={n}\n")

    # Which half of the criterion fails at the end?
    fail12 = np.array([(r["xy12"] >= XY_TH) or (r["z12"] >= 0) or (abs(r["z12"]) - H_DIFF >= H_TH) for r in rows])
    fail23 = np.array([(r["xy23"] >= XY_TH) or (r["z23"] >= 0) or (abs(r["z23"]) - H_DIFF >= H_TH) for r in rows])
    print("at the FINAL frame, which stack has broken:")
    print(f"  cube_1/cube_2 pair broken : {fail12.sum():3d} ({fail12.mean() * 100:5.1f}%)")
    print(f"  cube_2/cube_3 pair broken : {fail23.sum():3d} ({fail23.mean() * 100:5.1f}%)")
    print(f"  both broken               : {(fail12 & fail23).sum():3d}")

    # Is the top cube on the table? |z| near 0 rather than near the cube height.
    z23 = np.array([r["z23"] for r in rows])
    on_table3 = np.abs(z23) < 0.02
    inverted3 = z23 > 0.02
    print("\ncube_3 relative to cube_2 at the final frame:")
    print(f"  sitting at the same height (fell to the table): {on_table3.sum():3d} ({on_table3.mean() * 100:5.1f}%)")
    print(f"  BELOW cube_2 (stack order inverted)           : {inverted3.sum():3d} ({inverted3.mean() * 100:5.1f}%)")
    print(f"  still above cube_2                            : {(~on_table3 & ~inverted3).sum():3d}")

    xy23 = np.array([r["xy23"] for r in rows])
    print(f"\nfinal xy distance cube_2->cube_3 (threshold {XY_TH * 100:.0f} cm):")
    for q in (10, 50, 90):
        print(f"  p{q:<3d} = {np.percentile(xy23, q) * 100:6.2f} cm")

    drift3 = np.array([r["drift3"] for r in rows])
    print("\nhow far cube_3 moved between the last geometry frame and the episode end:")
    for q in (10, 50, 90):
        print(f"  p{q:<3d} = {np.percentile(drift3, q) * 100:6.2f} cm")

    # Split by whether a 3 mm tolerance would have "recovered" the episode.
    rec = best <= 3e-3
    print("\nsplit by whether a 3 mm gripper tolerance would have passed them:")
    for label, m in (("would pass at 3 mm", rec), ("still rejected at 3 mm", ~rec)):
        if m.sum() == 0:
            continue
        print(
            f"  {label:24s} n={m.sum():3d}  "
            f"median final xy23={np.median(xy23[m]) * 100:5.2f}cm  "
            f"median cube_3 drift={np.median(drift3[m]) * 100:5.2f}cm  "
            f"cube_3 fell to table={on_table3[m].mean() * 100:5.1f}%"
        )


if __name__ == "__main__":
    main()
