"""What kind of error is the residual in the frozen tail?

Two mechanisms are consistent with everything measured so far, and they call for different remedies.

  tracking lag -- the action is `target - current` scaled by 0.5, so the arm trails a moving target
      by a steady-state amount set by the target's speed and the effective gain. A configuration
      difference changes the effective gain of the damped-least-squares solve, so the two runs trail
      by different amounts. Signature: the residual points ALONG the direction of travel.

  configuration offset -- resolved-rate IK integrates joint velocities, and a closed path in
      Cartesian space does not close in joint space, so the arm sits in a different null-space
      configuration and the end-effector inherits a static offset. Signature: the residual points in
      an arbitrary direction, and correlates with the joint-space difference.

Both may contribute. The direction test separates them; the correlation test measures how much of
the end-effector residual the configuration difference accounts for.
"""

import sys

import h5py
import numpy as np

ROOT = "/home/pk/IsaacLab/datasets/perturbation_sweep_v2_arc"
CUBES = ("cube_1", "cube_2", "cube_3")
ARM = slice(0, 7)


def key_of(ro):
    return tuple(np.round(np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]), 6))


def load_run(seed, cm):
    tag = f"cm{cm:.1f}".replace(".", "p")
    out = {}
    for suffix, ok in (("", True), ("_failed", False)):
        with h5py.File(f"{ROOT}/seed{seed}/generated_{tag}{suffix}.hdf5", "r") as f:
            d = f["data"]
            for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
                out[key_of(d[k]["states"]["rigid_object"])] = {
                    "success": ok,
                    "eef": d[k]["obs"]["eef_pos"][:],
                    "joints": d[k]["obs"]["joint_pos"][:, ARM],
                }
    return out


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ref = load_run(seed, 0.5)
    frame = -5  # inside the frozen tail, a few frames before the end

    print(f"seed={seed}, paired against 0.5 cm, scenes succeeding at both, evaluated at frame {frame}\n")
    print(f"  {'cm':>5s} {'n':>5s} {'|residual|':>11s} {'|along v|':>11s} {'|across v|':>12s}"
          f" {'along share':>12s} {'corr(eef,joint)':>16s}")
    for cm in (1.0, 1.5, 2.0, 2.5, 3.0):
        other = load_run(seed, cm)
        mags, along, across, jnorm = [], [], [], []
        for k in ref:
            if k not in other or not (ref[k]["success"] and other[k]["success"]):
                continue
            e1, e2 = ref[k]["eef"], other[k]["eef"]
            if len(e1) < 8 or len(e2) < 8:
                continue
            r = e2[frame] - e1[frame]
            v = e1[frame] - e1[frame - 1]
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                continue
            vhat = v / nv
            par = float(np.dot(r, vhat))
            perp = float(np.linalg.norm(r - par * vhat))
            mags.append(float(np.linalg.norm(r)))
            along.append(abs(par))
            across.append(perp)
            jnorm.append(float(np.linalg.norm(ref[k]["joints"][frame] - other[k]["joints"][frame])))
        mags, along, across, jnorm = map(np.array, (mags, along, across, jnorm))
        # share of the squared residual that lies along the direction of travel
        share = (along**2).sum() / (mags**2).sum()
        corr = float(np.corrcoef(mags, jnorm)[0, 1])
        print(
            f"  {cm:5.1f} {len(mags):5d} {np.median(mags) * 100:10.3f}cm {np.median(along) * 100:10.3f}cm"
            f" {np.median(across) * 100:11.3f}cm {share * 100:11.1f}% {corr:16.3f}"
        )

    print("\n  a residual dominated by the 'across' component, uncorrelated with the direction of")
    print("  travel, is a configuration offset rather than trailing behind a moving target.")
    print("  isotropic reference: a random 3-D direction puts 1/3 of its squared length along v.")


if __name__ == "__main__":
    main()
