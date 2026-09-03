"""Which of the three proposed fixes has a premise that survives the data?

Each direction assumes a different quantity is the one that breaks the grasp:

  direction 1 (copy the joint configuration from a parallel unperturbed run) assumes the JOINT
      configuration difference is what does the damage.
  direction 2 (pull the gripper back to where it should be) assumes the gripper POSE difference is.
  direction 3 (hold the target still so the loop can close) assumes neither is fundamental -- that
      whatever the difference is, it persists only because the arm is chasing a moving target.

Directions 1 and 2 make opposite predictions that the existing sweep data can already settle,
without running anything: pair each arc episode with its unperturbed counterpart, measure both
quantities at the moment the grasp actually happens, and see which one separates the grasps that
worked from the ones that did not.

The grasp moment is found from the data rather than assumed: the first frame where the gripper
leaves its open value by more than a millimetre. That is the end of subtask 1, which is where the
fastest-growing failure mode ("cube_2 never moved") lives.
"""

import sys

import h5py
import numpy as np

ROOT = "/home/pk/IsaacLab/datasets/perturbation_sweep_v2_arc"
CUBES = ("cube_1", "cube_2", "cube_3")
OPEN_VAL = 0.04


def key_of(ro):
    return tuple(np.round(np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]), 6))


def grasp_frame(gripper_pos):
    """First frame where the jaws have closed appreciably; None if they never do."""
    jaw = np.minimum(gripper_pos[:, 0], -gripper_pos[:, 1])
    closed = np.where(jaw < OPEN_VAL - 0.001)[0]
    return int(closed[0]) if len(closed) else None


def load_run(seed, cm):
    tag = f"cm{cm:.1f}".replace(".", "p")
    out = {}
    for suffix, ok in (("", True), ("_failed", False)):
        with h5py.File(f"{ROOT}/seed{seed}/generated_{tag}{suffix}.hdf5", "r") as f:
            d = f["data"]
            for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
                obs = d[k]["obs"]
                ro = d[k]["states"]["rigid_object"]
                out[key_of(ro)] = {
                    "success": ok,
                    "pos": obs["eef_pos"][:],
                    "joints": obs["joint_pos"][:, :7],
                    "grip": obs["gripper_pos"][:],
                    "cube2": ro["cube_2"]["root_pose"][:, :3],
                }
    return out


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cm = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    ref = load_run(seed, 0.5)
    arc = load_run(seed, cm)

    rows = []
    for k, a in arc.items():
        r = ref.get(k)
        if r is None:
            continue
        ga, gr = grasp_frame(a["grip"]), grasp_frame(r["grip"])
        if ga is None or gr is None:
            continue
        # did the arc run actually pick cube_2 up?
        picked = float(np.linalg.norm(a["cube2"][-1] - a["cube2"][0])) > 0.02
        rows.append(
            {
                "picked": picked,
                "success": a["success"],
                "dpos": float(np.linalg.norm(a["pos"][ga] - r["pos"][gr])),
                "djoint": float(np.linalg.norm(a["joints"][ga] - r["joints"][gr])),
                # how far the gripper was from the cube it was trying to take
                "reach": float(np.linalg.norm(a["pos"][ga] - a["cube2"][ga])),
                "reach_ref": float(np.linalg.norm(r["pos"][gr] - r["cube2"][gr])),
            }
        )

    picked = np.array([r["picked"] for r in rows])
    succ = np.array([r["success"] for r in rows])
    dpos = np.array([r["dpos"] for r in rows])
    djoint = np.array([r["djoint"] for r in rows])
    reach = np.array([r["reach"] for r in rows])
    reach_ref = np.array([r["reach_ref"] for r in rows])

    print(f"seed={seed}  arc={cm}cm  paired episodes with a grasp in both: {len(rows)}")
    print(f"  cube_2 picked up: {picked.sum()}/{len(rows)}   full episode success: {succ.sum()}\n")

    print("=== at the grasp frame, split by whether the grasp worked ===")
    print(f"  {'quantity':<34s} {'picked up':>12s} {'not picked':>12s} {'separation':>11s}")
    for name, v, unit, scale in (
        ("gripper position vs unperturbed", dpos, "cm", 100),
        ("joint configuration vs unperturbed", djoint, "rad", 1),
        ("gripper-to-cube distance (arc)", reach, "cm", 100),
        ("gripper-to-cube distance (unperturbed)", reach_ref, "cm", 100),
    ):
        a, b = v[picked], v[~picked]
        if len(b) == 0:
            print(f"  {name:<34s} {np.median(a) * scale:11.3f}{unit}  (no failures to compare)")
            continue
        # Cohen's d as a scale-free measure of how well the quantity separates the two groups
        pooled = np.sqrt((a.var() + b.var()) / 2) or 1e-12
        d = (b.mean() - a.mean()) / pooled
        print(
            f"  {name:<34s} {np.median(a) * scale:11.3f}{unit} {np.median(b) * scale:11.3f}{unit}"
            f" {d:11.2f}"
        )
    print("\n  'separation' is Cohen's d: how many pooled standard deviations apart the two groups")
    print("  are. Below ~0.2 the quantity carries essentially no information about the outcome.")


if __name__ == "__main__":
    main()
