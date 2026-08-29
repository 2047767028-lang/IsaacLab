"""Measure the in-flight-success defect rate in a freshly generated dataset.

For each ACCEPTED demo, ask whether the stack is still standing at the last frame. An episode where
the criterion fired on a cube passing through the stacked configuration ends with that cube on the
table, so this counts the defect directly rather than inferring it.

Usage:  python analyze_runs.py <run.hdf5> [<run.hdf5> ...]
"""

import sys

import h5py
import numpy as np

XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
DT = 0.05


def stacked_geom(cu):
    c1, c2, c3 = cu[:, 0:3], cu[:, 3:6], cu[:, 6:9]
    out = None
    for a, b in ((c1, c2), (c2, c3)):
        dd = a - b
        ok = (np.linalg.norm(dd[:, :2], axis=1) < XY_TH)
        ok &= (np.linalg.norm(dd[:, 2:], axis=1) - H_DIFF < H_TH) & (dd[:, 2] < 0.0)
        out = ok if out is None else (out & ok)
    return out


def run(path):
    with h5py.File(path, "r") as f:
        d = f["data"]
        keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
        broken, speeds, xy_end = [], [], []
        for k in keys:
            cu = d[k]["obs"]["cube_positions"][:]
            g = stacked_geom(cu)
            broken.append(not bool(g[-1]))
            v = np.gradient(cu[:, 8], DT)
            speeds.append(float(np.min(np.abs(v[g]))) if g.any() else np.nan)
            xy_end.append(float(np.linalg.norm((cu[-1, 3:6] - cu[-1, 6:9])[:2])))
    n = len(keys)
    b = np.array(broken)
    name = path.split("/")[-1]
    print(f"{name:14s} accepted={n:4d}  ends broken={b.sum():3d} ({b.mean() * 100:5.2f}%)", end="")
    if b.any():
        print(
            f"   median cube_3 speed at the qualifying frame: broken={np.nanmedian(np.array(speeds)[b]):.4f}"
            f" vs intact={np.nanmedian(np.array(speeds)[~b]):.4f} m/s"
            f"   median final xy={np.median(np.array(xy_end)[b]) * 100:.2f} cm"
        )
    else:
        print("   (no defects)")
    return n, int(b.sum())


def main():
    for p in sys.argv[1:]:
        try:
            run(p)
        except Exception as e:  # a run that produced no file at all should say so, not crash the rest
            print(f"{p}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
