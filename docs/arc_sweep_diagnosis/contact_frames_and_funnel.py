"""(A) One source demo, frame by frame around the first grasp: what is actually recorded.
(B) Stage funnel for the contact-hold runs: where does the dwell help?"""
import h5py
import numpy as np

SRC = "/home/pk/IsaacLab/datasets/annotated_dataset.hdf5"
OUT = "/home/pk/IsaacLab/datasets/arc_sweep_diagnosis_runs/contact_hold"
OPEN_VAL = 0.04


def transitions(grip):
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = jaw < OPEN_VAL - 0.001
    return np.where(np.diff(closed.astype(int)) != 0)[0] + 1


print("(A) demo_0, frames around the first jaw closure. Recorded fields only.")
print(f"  {'frame':>5s} {'|action xyz| = commanded step':>30s} {'gripper cmd':>11s} {'arm moved since prev':>21s} "
      f"{'jaw opening':>11s} {'eef-cube_2':>10s}")
with h5py.File(SRC, "r") as f:
    d = f["data/demo_0"]
    act = d["actions"][:]
    pos = d["obs/eef_pos"][:]
    grip = d["obs/gripper_pos"][:]
    c2 = d["states/rigid_object/cube_2/root_pose"][:, :3]
    tg = d["obs/datagen_info/target_eef_pose"]
    tgt = tg[list(tg.keys())[0]][:] if isinstance(tg, h5py.Group) else tg[:]
    g = transitions(grip)[0]
    for t in range(g - 8, g + 3):
        step = np.linalg.norm(pos[t] - pos[t - 1]) * 100
        jaw = min(grip[t, 0], -grip[t, 1]) * 100
        print(f"  {t:5d} {np.linalg.norm(act[t, :3]) * 100:26.2f} cm {act[t, -1]:+9.0f}   {step:15.3f} cm"
              f" {jaw:9.2f} cm {np.linalg.norm(pos[t] - c2[t]) * 100:8.2f} cm")
    # the stored target_eef_pose IS current + action: check once
    err = np.linalg.norm(tgt[:, :3, 3] - (pos + act[:, :3]), axis=1).max() * 100 if tgt.ndim == 3 else float("nan")
    print(f"  check: stored target_eef_pose == eef_pos + action[:3] to within {err:.4f} cm over the whole demo")

print()
print("(B) stage funnel per run (fraction of all 300 attempts reaching each stage)")
print(f"  {'run':<10s} {'grasp1':>8s} {'place1':>8s} {'grasp2':>8s} {'success':>8s}")
for tag in ("ref_none", "ref_hold", "arc_none", "arc_snap", "arc_hold"):
    n = g1 = p1 = g2 = ok = 0
    for suffix, succ in (("", True), ("_failed", False)):
        with h5py.File(f"{OUT}/ch_{tag}{suffix}.hdf5", "r") as f:
            for k in f["data"]:
                ro = f["data"][k]["states"]["rigid_object"]
                c1 = ro["cube_1"]["root_pose"][:, :3]
                c2 = ro["cube_2"]["root_pose"][:, :3]
                c3 = ro["cube_3"]["root_pose"][:, :3]
                n += 1
                a = np.linalg.norm(c2[-1] - c2[0]) > 0.02
                b = a and np.linalg.norm((c2[-1] - c1[-1])[:2]) < 0.02 and (c2[-1, 2] - c1[-1, 2]) > 0.03
                c = b and np.linalg.norm(c3[-1] - c3[0]) > 0.02
                g1 += a; p1 += b; g2 += c; ok += succ
    print(f"  {tag:<10s} {100*g1/n:7.1f}% {100*p1/n:7.1f}% {100*g2/n:7.1f}% {100*ok/n:7.1f}%")
print("  grasp1: cube_2 moved >2cm; place1: cube_2 ends on cube_1; grasp2: and cube_3 moved; success: task criterion")
