"""Why do geometry and gripper-open never coincide in the rejected episodes?

The settling check turned up something that rules out the first diagnosis: in all 76 rejected
episodes the jaws DO reach fully open (final jaw error ~0.00 mm), yet at every frame where the
stack geometry held, the jaws were still far from open. Geometry and release never overlap.

Two candidate explanations, with opposite implications:
  (a) the stack is disturbed by the release -- the cube shifts when the jaws let go, so geometry
      breaks exactly as the gripper opens. That is a genuine failure and the criterion is right.
  (b) the geometry window and the release window simply sit at different times for some other
      reason, e.g. the cube is still settling and drifts back into tolerance after the episode ends.

This prints the actual timelines so the two can be told apart.
"""

import h5py
import numpy as np

OPEN_VAL = 0.04
XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
DEFAULT_TOL = 1e-4 + 1e-4 * OPEN_VAL


def geom_terms(cu):
    c1, c2, c3 = cu[:, 0:3], cu[:, 3:6], cu[:, 6:9]
    d12, d23 = c1 - c2, c2 - c3
    return {
        "xy12": np.linalg.norm(d12[:, :2], axis=1),
        "xy23": np.linalg.norm(d23[:, :2], axis=1),
        "h12": np.linalg.norm(d12[:, 2:], axis=1),
        "h23": np.linalg.norm(d23[:, 2:], axis=1),
        "z12": d12[:, 2],
        "z23": d23[:, 2],
    }


def stacked_geom(cu):
    t = geom_terms(cu)
    ok = (t["xy12"] < XY_TH) & (t["xy23"] < XY_TH)
    ok &= (t["h12"] - H_DIFF < H_TH) & (t["z12"] < 0.0)
    ok &= (t["h23"] - H_DIFF < H_TH) & (t["z23"] < 0.0)
    return ok


def jaw_error(gp):
    return np.maximum(np.abs(gp[:, 0] - OPEN_VAL), np.abs(-gp[:, 1] - OPEN_VAL))


def main():
    path = "/home/pk/IsaacLab/datasets/pi05_training_data_v1/baseline/generated_failed.hdf5"
    f = h5py.File(path, "r")
    d = f["data"]
    keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))

    picked, summary = [], []
    for k in keys:
        cu = d[k]["obs"]["cube_positions"][:]
        gp = d[k]["obs"]["gripper_pos"][:]
        g = stacked_geom(cu)
        if not g.any():
            continue
        err = jaw_error(gp)
        if err[g].min() <= DEFAULT_TOL:
            continue
        gi = np.where(g)[0]
        oi = np.where(err <= DEFAULT_TOL)[0]
        summary.append(
            {
                "k": k,
                "T": len(g),
                "geom_first": int(gi[0]),
                "geom_last": int(gi[-1]),
                "geom_n": len(gi),
                "open_first": int(oi[0]) if len(oi) else -1,
                "geom_at_end": bool(g[-1]),
            }
        )
        picked.append((k, cu, gp, g, err))

    print(f"rejected-with-geometry episodes: N={len(summary)}\n")
    geom_at_end = np.array([s["geom_at_end"] for s in summary])
    print(f"geometry still holds at the final frame : {geom_at_end.sum():3d} ({geom_at_end.mean() * 100:5.1f}%)")
    ever_open = np.array([s["open_first"] >= 0 for s in summary])
    print(f"jaws reach the open tolerance at all    : {ever_open.sum():3d} ({ever_open.mean() * 100:5.1f}%)")

    gl = np.array([s["geom_last"] for s in summary])
    of = np.array([s["open_first"] for s in summary])
    both = ever_open
    lag = of[both] - gl[both]
    print(f"\nframes between the last geometry frame and the first open frame (positive = open came later):")
    for q in (10, 50, 90):
        print(f"  p{q:<3d} = {np.percentile(lag, q):+7.1f}")
    print(f"  min={lag.min():+d}  max={lag.max():+d}")

    gn = np.array([s["geom_n"] for s in summary])
    print(f"\nhow many frames the geometry held at all: median={np.median(gn):.0f}  min={gn.min()}  max={gn.max()}")

    print("\ntimeline for the four worst cases:")
    for k, cu, gp, g, err in picked[:0] or []:
        pass
    order = sorted(range(len(picked)), key=lambda i: -jaw_error(picked[i][2])[picked[i][3]].min())[:4]
    for i in order:
        k, cu, gp, g, err = picked[i]
        t = geom_terms(cu)
        gi = np.where(g)[0]
        print(f"\n  {k}: T={len(g)}  geometry frames {gi[0]}..{gi[-1]} ({len(gi)} frames)")
        marks = sorted(set([0, gi[0], gi[len(gi) // 2], gi[-1], min(gi[-1] + 3, len(g) - 1), len(g) - 1]))
        print(f"    {'frame':>6s} {'geom':>5s} {'xy23':>8s} {'h23-Δ':>8s} {'z23':>8s} {'jawerr':>9s}")
        for fr in marks:
            print(
                f"    {fr:6d} {str(bool(g[fr])):>5s} {t['xy23'][fr] * 100:7.2f}cm "
                f"{(t['h23'][fr] - H_DIFF) * 100:7.2f}cm {t['z23'][fr] * 100:7.2f}cm {err[fr] * 1000:8.3f}mm"
            )
    f.close()


if __name__ == "__main__":
    main()
