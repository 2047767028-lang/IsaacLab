"""Are the gripper-rejected episodes still opening when the episode ends, or are they stuck?

This is the load-bearing question for whether relaxing the tolerance is a legitimate fix or just a
looser criterion that lets real failures through. Two very different states produce a jaw that is
not at the open value:

  (a) the jaws are travelling toward open and the episode ran out of steps -- the object is already
      released, the task is physically complete, and the criterion is rejecting it on timing alone;
  (b) the jaws are still closed around the object -- a genuine failure that no tolerance should pass.

They are separable: in (a) the jaw error shrinks over the closing frames and the final opening is
already far wider than the cube; in (b) it plateaus at roughly the cube's half-width.

Franka cube side = 4.68 cm, so a jaw holding the cube sits near 0.0234 m and the open value is
0.04 m -- a jaw error near 0.0166 m means "still holding".
"""

import h5py
import numpy as np

OPEN_VAL = 0.04
CUBE = 0.0468
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
    path = "/home/pk/IsaacLab/datasets/pi05_training_data_v1/baseline/generated_failed.hdf5"
    f = h5py.File(path, "r")
    d = f["data"]
    keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))

    rows = []
    for k in keys:
        cu = d[k]["obs"]["cube_positions"][:]
        gp = d[k]["obs"]["gripper_pos"][:]
        g = stacked_geom(cu)
        if not g.any():
            continue
        err = jaw_error(gp)
        best = err[g].min()
        if best <= DEFAULT_TOL:
            continue  # would already have passed
        # Trend over the final frames: negative slope = jaws still opening when the episode ended.
        tail = err[-10:]
        slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) >= 3 else np.nan
        rows.append((k, best, float(err[-1]), slope, float(min(gp[-1, 0], -gp[-1, 1]))))
    f.close()

    best = np.array([r[1] for r in rows])
    final = np.array([r[2] for r in rows])
    slope = np.array([r[3] for r in rows])
    jaw_final = np.array([r[4] for r in rows])

    print(f"gripper-rejected episodes (geometry already satisfied): N={len(rows)}\n")

    still_opening = slope < -1e-6
    print(f"jaw error still shrinking over the last 10 frames : {still_opening.sum():3d} ({still_opening.mean() * 100:5.1f}%)")
    print(f"jaw error flat or growing                         : {(~still_opening).sum():3d} ({(~still_opening).mean() * 100:5.1f}%)\n")

    # "Released" is a physical question, not a tolerance question: is the opening already wider than
    # the cube it was holding?
    released = jaw_final * 2 > CUBE
    print(f"final opening already wider than the cube (released): {released.sum():3d} ({released.mean() * 100:5.1f}%)")
    print(f"final opening still narrower than the cube (holding): {(~released).sum():3d} ({(~released).mean() * 100:5.1f}%)\n")

    print("cross-tab, by whether a 3 mm tolerance would recover the episode:")
    rec = best <= 3e-3
    for label, mask in (("recovered at 3 mm", rec), ("still rejected at 3 mm", ~rec)):
        if mask.sum() == 0:
            continue
        print(
            f"  {label:24s} n={mask.sum():3d}  "
            f"released={released[mask].mean() * 100:5.1f}%  "
            f"still-opening={still_opening[mask].mean() * 100:5.1f}%  "
            f"median final jaw error={np.median(final[mask]) * 1000:6.2f} mm"
        )

    print("\nworst offenders (largest best-case jaw error), to see what the tail actually is:")
    order = np.argsort(-best)[:8]
    print(f"  {'demo':10s} {'best err':>9s} {'final err':>10s} {'final jaw':>10s} {'opening':>9s} {'released':>9s}")
    for i in order:
        k, b, fe, _, jf = rows[i]
        print(f"  {k:10s} {b * 1000:8.2f}mm {fe * 1000:9.2f}mm {jf:10.4f} {jf * 2 * 100:8.2f}cm {str(jf * 2 > CUBE):>9s}")


if __name__ == "__main__":
    main()
