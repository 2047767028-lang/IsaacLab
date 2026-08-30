"""Is the residual at the contact phase tracking lag, or null-space drift?

The paired test showed that a constant ~45% of the arc amplitude survives into the frozen tail as
end-effector deviation, even though the target there is byte-identical between the two runs. Two
mechanisms produce that, and they call for different remedies:

  tracking lag   -- the IK-Rel controller applies a scaled fraction of each commanded delta, so the
                    arm trails a moving target. A longer frozen tail lets it catch up.
  null-space drift -- a 7-DOF arm reaching the same end-effector pose through a different path can
                    settle into a different elbow configuration, which does not decay just because
                    the end-effector target came back. A longer tail does not fix that.

They separate on how the arm-joint deviation compares to the end-effector deviation, and on whether
the deviation shrinks across the tail. Under lag, both decay together toward the end of the episode.
Under drift, the joints stay apart while the end-effector converges.

Restricted to scenes succeeding at both amplitudes, so nothing here is a selection effect.
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

    print(f"seed={seed}, paired against 0.5 cm, scenes succeeding at both\n")
    print("=== deviation by distance from the end of the episode ===")
    print(f"  {'cm':>5s} {'n':>4s} " + " ".join(f"{f'eef@-{w}':>10s}" for w in (30, 20, 10, 1)) + "   " + " ".join(f"{f'joint@-{w}':>11s}" for w in (30, 20, 10, 1)))
    for cm in (1.0, 2.0, 3.0):
        other = load_run(seed, cm)
        eef_by_w = {w: [] for w in (30, 20, 10, 1)}
        jnt_by_w = {w: [] for w in (30, 20, 10, 1)}
        n = 0
        for k in ref:
            if k not in other or not (ref[k]["success"] and other[k]["success"]):
                continue
            n += 1
            for w in (30, 20, 10, 1):
                i1, i2 = len(ref[k]["eef"]) - w, len(other[k]["eef"]) - w
                if i1 < 0 or i2 < 0:
                    continue
                eef_by_w[w].append(float(np.linalg.norm(ref[k]["eef"][i1] - other[k]["eef"][i2])))
                jnt_by_w[w].append(float(np.linalg.norm(ref[k]["joints"][i1] - other[k]["joints"][i2])))
        print(
            f"  {cm:5.1f} {n:4d} "
            + " ".join(f"{np.median(eef_by_w[w]) * 100:9.3f}cm" for w in (30, 20, 10, 1))
            + "   "
            + " ".join(f"{np.median(jnt_by_w[w]):8.4f}rad" for w in (30, 20, 10, 1))
        )

    print("\n  reading: if both columns shrink toward the end, the residual is lag that a longer")
    print("  frozen tail would absorb; if the joint column stays flat while eef shrinks, the arm")
    print("  settled into a different null-space configuration and a longer tail will not help.")


if __name__ == "__main__":
    main()
