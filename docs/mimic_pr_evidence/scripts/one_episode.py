"""Full timeline of one rejected episode, to settle what the geometry window actually is.

Two readings are still open after the aggregate numbers:
  (a) the gripper released cube_3 onto cube_2 and it fell -- a real failure;
  (b) the geometry window is coincidental (e.g. cube_3 passing over cube_2 while carried), in which
      case the episode never even attempted a placement there and the aggregate is misleading.

They differ in whether cube_3 is descending onto cube_2 and coming to rest, or sweeping through.
Printing the vertical gap, the horizontal offset and the jaw over the whole window separates them.
"""

import sys

import h5py
import numpy as np

OPEN_VAL = 0.04
XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
DEFAULT_TOL = 1e-4 + 1e-4 * OPEN_VAL


def stacked_geom(cu):
    c1, c2, c3 = cu[:, 0:3], cu[:, 3:6], cu[:, 6:9]
    d12, d23 = c1 - c2, c2 - c3
    ok = (np.linalg.norm(d12[:, :2], axis=1) < XY_TH) & (np.linalg.norm(d23[:, :2], axis=1) < XY_TH)
    ok &= (np.linalg.norm(d12[:, 2:], axis=1) - H_DIFF < H_TH) & (d12[:, 2] < 0.0)
    ok &= (np.linalg.norm(d23[:, 2:], axis=1) - H_DIFF < H_TH) & (d23[:, 2] < 0.0)
    return ok


def jaw_error(gp):
    return np.maximum(np.abs(gp[:, 0] - OPEN_VAL), np.abs(-gp[:, 1] - OPEN_VAL))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "best3mm"
    path = "/home/pk/IsaacLab/datasets/pi05_training_data_v1/baseline/generated_failed.hdf5"
    f = h5py.File(path, "r")
    d = f["data"]

    cands = []
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
        cands.append((best, k))
    cands.sort()
    # The most favourable case for the "tolerance too tight" reading: the one whose gripper came
    # closest to open while the geometry held.
    best, key = cands[0]
    print(f"episode {key}: closest the jaws came to open while stacked = {best * 1000:.3f} mm\n")

    cu = d[key]["obs"]["cube_positions"][:]
    gp = d[key]["obs"]["gripper_pos"][:]
    ee = d[key]["obs"]["eef_pos"][:]
    g = stacked_geom(cu)
    err = jaw_error(gp)
    gi = np.where(g)[0]
    lo, hi = max(0, gi[0] - 8), min(len(g) - 1, gi[-1] + 25)

    c2, c3 = cu[:, 3:6], cu[:, 6:9]
    dz = c3[:, 2] - c2[:, 2]
    dxy = np.linalg.norm((c3 - c2)[:, :2], axis=1)

    print(f"  T={len(g)}   geometry frames {gi[0]}..{gi[-1]}   window shown {lo}..{hi}")
    print(f"  {'frame':>6s} {'geom':>5s} {'c3-c2 dz':>9s} {'c3-c2 xy':>9s} {'c3 z':>8s} {'jaw err':>9s} {'eef z':>8s}")
    for fr in range(lo, hi + 1):
        star = " *" if g[fr] else "  "
        print(
            f"  {fr:6d}{star}{'':>3s} {dz[fr] * 100:8.2f}cm {dxy[fr] * 100:8.2f}cm "
            f"{c3[fr, 2] * 100:7.2f}cm {err[fr] * 1000:8.3f}mm {ee[fr, 2] * 100:7.2f}cm"
        )
    print(f"\n  final frame {len(g) - 1}: c3-c2 dz={dz[-1] * 100:.2f}cm  xy={dxy[-1] * 100:.2f}cm  "
          f"c3 z={c3[-1, 2] * 100:.2f}cm  jaw err={err[-1] * 1000:.3f}mm")
    f.close()


if __name__ == "__main__":
    main()
