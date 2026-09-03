"""Does the gripper's orientation drift too, or only its position?

What decides whether a parallel gripper closes on a cube is the gripper's pose -- position and
orientation. Where the elbow sits matters only if it collides with something. So the size of the
joint-configuration difference is not itself the quantity of interest: what matters is how much of
it shows up at the gripper.

If the orientation is intact and only the position has moved, then freezing the joint configuration
buys nothing that freezing the end-effector pose would not, and the joint offset is largely
null-space -- real, but invisible where it counts.

Measured on the same paired, both-succeed sample as everything else.
"""

import sys

import h5py
import numpy as np

ROOT = "/home/pk/IsaacLab/datasets/perturbation_sweep_v2_arc"
CUBES = ("cube_1", "cube_2", "cube_3")


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
                    "pos": d[k]["obs"]["eef_pos"][:],
                    "quat": d[k]["obs"]["eef_quat"][:],
                    "joints": d[k]["obs"]["joint_pos"][:, :7],
                }
    return out


def quat_angle(q1, q2):
    """Smallest rotation angle between two unit quaternions, in degrees. Sign-agnostic."""
    d = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return float(np.degrees(2 * np.arccos(np.clip(d, -1.0, 1.0))))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ref = load_run(seed, 0.5)
    frame = -5

    print(f"seed={seed}, paired against 0.5 cm, scenes succeeding at both, frame {frame}\n")
    print(f"  {'cm':>5s} {'n':>5s} {'position':>10s} {'orientation':>13s} {'joint norm':>12s}"
          f" {'per-joint max':>14s}")
    for cm in (1.0, 1.5, 2.0, 2.5, 3.0):
        other = load_run(seed, cm)
        dp, da, dj, dmax = [], [], [], []
        for k in ref:
            if k not in other or not (ref[k]["success"] and other[k]["success"]):
                continue
            a, b = ref[k], other[k]
            if len(a["pos"]) < 8 or len(b["pos"]) < 8:
                continue
            dp.append(float(np.linalg.norm(a["pos"][frame] - b["pos"][frame])))
            da.append(quat_angle(a["quat"][frame], b["quat"][frame]))
            diff = a["joints"][frame] - b["joints"][frame]
            dj.append(float(np.linalg.norm(diff)))
            dmax.append(float(np.max(np.abs(diff))))
        print(
            f"  {cm:5.1f} {len(dp):5d} {np.median(dp) * 100:9.3f}cm {np.median(da):12.3f}deg"
            f" {np.median(dj):11.4f}rad {np.degrees(np.median(dmax)):13.2f}deg"
        )

    print("\n  For scale: the cube is 4.68 cm and the gripper's jaws travel 8 cm fully open, so a")
    print("  centimetre of position error is a large fraction of the grasp tolerance, while a few")
    print("  degrees of gripper rotation is not.")


if __name__ == "__main__":
    main()
