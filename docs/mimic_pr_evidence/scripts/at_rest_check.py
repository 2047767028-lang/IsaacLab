"""Is the top cube at rest when the stack criterion fires, or passing through in free fall?

`cubes_stacked` tests an instantaneous geometric configuration: cube_3 within 4 cm of cube_2 in xy
and one cube-height above it. A cube dropped from above cube_2 satisfies exactly that description
for a frame or two on its way down. Nothing in the criterion requires the cube to be supported or
stationary.

This measures cube_3's vertical speed at the frames where the geometry holds, in two populations:
  - the 76 episodes rejected only by the gripper tolerance (the ones a relaxed tolerance would admit)
  - the accepted successes (which the same blind spot could also be letting through)

Frames are 20 Hz (env dt 0.01 s, decimation 5), so one frame is 0.05 s. Free fall gains 0.49 m/s in
that time; a cube resting on another cube reads ~0.
"""

import h5py
import numpy as np

OPEN_VAL = 0.04
XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
DEFAULT_TOL = 1e-4 + 1e-4 * OPEN_VAL
DT = 0.05
BASE = "/home/pk/IsaacLab/datasets/pi05_training_data_v1/baseline"


def stacked_geom(cu):
    c1, c2, c3 = cu[:, 0:3], cu[:, 3:6], cu[:, 6:9]
    d12, d23 = c1 - c2, c2 - c3
    ok = (np.linalg.norm(d12[:, :2], axis=1) < XY_TH) & (np.linalg.norm(d23[:, :2], axis=1) < XY_TH)
    ok &= (np.linalg.norm(d12[:, 2:], axis=1) - H_DIFF < H_TH) & (d12[:, 2] < 0.0)
    ok &= (np.linalg.norm(d23[:, 2:], axis=1) - H_DIFF < H_TH) & (d23[:, 2] < 0.0)
    return ok


def jaw_error(gp):
    return np.maximum(np.abs(gp[:, 0] - OPEN_VAL), np.abs(-gp[:, 1] - OPEN_VAL))


def vz(cu):
    """cube_3 vertical speed, m/s, central difference, same length as the trajectory."""
    z = cu[:, 8]
    v = np.gradient(z, DT)
    return v


def collect(path, want_rejected: bool):
    f = h5py.File(path, "r")
    d = f["data"]
    out = []
    for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
        cu = d[k]["obs"]["cube_positions"][:]
        gp = d[k]["obs"]["gripper_pos"][:]
        g = stacked_geom(cu)
        if not g.any():
            continue
        err = jaw_error(gp)
        best = err[g].min()
        if want_rejected and best <= DEFAULT_TOL:
            continue
        v = vz(cu)
        if want_rejected:
            frames = np.where(g)[0]
        else:
            # the frames that actually made it a success: geometry AND gripper open
            frames = np.where(g & (err <= DEFAULT_TOL))[0]
            if len(frames) == 0:
                continue
        # the most favourable frame for the episode: whichever qualifying frame is most at rest
        i = frames[np.argmin(np.abs(v[frames]))]
        out.append({"k": k, "vz": float(v[i]), "z3": float(cu[i, 8]), "frame": int(i), "T": len(g)})
    f.close()
    return out


def report(name, rows):
    v = np.abs(np.array([r["vz"] for r in rows]))
    print(f"\n{name}: N={len(rows)}")
    print("  |cube_3 vertical speed| at its most-at-rest qualifying frame (m/s):")
    for q in (10, 50, 90):
        print(f"    p{q:<3d} = {np.percentile(v, q):.4f}")
    print(f"    max  = {v.max():.4f}")
    for thr in (0.01, 0.05, 0.10):
        print(f"    share moving faster than {thr:.2f} m/s: {(v > thr).mean() * 100:5.1f}%")
    return v


def main():
    rej = collect(f"{BASE}/generated_failed.hdf5", want_rejected=True)
    acc = collect(f"{BASE}/generated.hdf5", want_rejected=False)
    v_rej = report("rejected only by the gripper tolerance", rej)
    v_acc = report("accepted successes", acc)

    print("\nseparation:")
    print(f"  rejected  median {np.median(v_rej):.4f} m/s")
    print(f"  accepted  median {np.median(v_acc):.4f} m/s")
    thr = 0.02
    print(f"\n  at a 'moving faster than {thr} m/s' cut:")
    print(f"    rejected classified as in-flight : {(v_rej > thr).mean() * 100:5.1f}%")
    print(f"    accepted classified as in-flight : {(v_acc > thr).mean() * 100:5.1f}%")


if __name__ == "__main__":
    main()
