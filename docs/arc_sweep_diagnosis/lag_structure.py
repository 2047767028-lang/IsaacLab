"""Is the "arm trails its target by 5 cm" statement true at every frame, or only while moving?

Three measurements, all on data already on disk:

(1) Source demos: the distribution of |target - achieved| (= |actions[:, :3]|, since
    action_to_target_eef_pose sets target = current + action, unscaled), split into "the gripper is
    about to act" frames and all other frames.

(2) Source demos: where the subtask termination signal fires relative to the gripper transition.
    If the signal fires AT the transition, then adding subtask_term_offset_range=(10,20) puts the
    subtask boundary 10-20 frames after the gripper has already acted.

(3) Generated runs: how far the arm travels in the 20 frames before the gripper transition, with
    and without the inserted dwell. A dwell that works shows the arm nearly stationary at contact.
"""
import h5py
import numpy as np

SRC = "/home/pk/IsaacLab/datasets/annotated_dataset.hdf5"
OUT = "/home/pk/IsaacLab/datasets/arc_sweep_diagnosis_runs/contact_hold"
OPEN_VAL = 0.04


def transitions(grip):
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = jaw < OPEN_VAL - 0.001
    return np.where(np.diff(closed.astype(int)) != 0)[0] + 1


print("=" * 78)
print("(1) SOURCE DEMOS: is the 5 cm lag present at every frame?")
print("=" * 78)
near, far, at_contact = [], [], []
with h5py.File(SRC, "r") as f:
    for k in f["data"]:
        d = f["data"][k]
        lag = np.linalg.norm(d["actions"][:, :3], axis=1) * 100
        ev = transitions(d["obs/gripper_pos"][:])
        mask = np.zeros(len(lag), bool)
        for e in ev:
            mask[max(0, e - 3):e + 1] = True     # the gripper-acts window
            at_contact.append(lag[e])
        near.append(lag[mask])
        far.append(lag[~mask])
near, far, at_contact = np.concatenate(near), np.concatenate(far), np.array(at_contact)
for name, v in (("within 3 frames of a gripper action", near), ("all other frames", far),
                ("exactly at the gripper action", at_contact)):
    print(f"  {name:<38s} median {np.median(v):5.2f} cm   p90 {np.percentile(v, 90):6.2f} cm   "
          f"frac < 0.3 cm: {100 * (v < 0.3).mean():4.1f}%   (n={len(v)})")

print()
print("=" * 78)
print("(2) SOURCE DEMOS: subtask termination signal vs the gripper action")
print("=" * 78)
with h5py.File(SRC, "r") as f:
    d0 = f["data/demo_0"]
    sig_names = list(d0["datagen_info/subtask_term_signals"].keys()) if "datagen_info" in d0 else []
    print(f"  signals present: {sig_names}")
    offs = {s: [] for s in sig_names}
    for k in f["data"]:
        d = f["data"][k]
        ev = transitions(d["obs/gripper_pos"][:])
        for s in sig_names:
            sig = np.asarray(d[f"datagen_info/subtask_term_signals/{s}"]).reshape(-1)
            fired = np.where(sig > 0.5)[0]
            if not len(fired):
                continue
            t = int(fired[0])
            offs[s].append(min((t - e for e in ev), key=abs))
    for s in sig_names:
        v = np.array(offs[s])
        if len(v):
            print(f"  {s:<16s} signal fires {np.median(v):+.1f} frames from the nearest gripper action "
                  f"(min {v.min():+d}, max {v.max():+d}, n={len(v)})")
print("  -> the subtask boundary is this + a sampled subtask_term_offset_range of (10, 20) frames")

print()
print("=" * 78)
print("(3) GENERATED: arm travel in the 20 frames before the gripper acts")
print("=" * 78)
print(f"  {'run':<10s} {'travel over last 20 frames':>28s} {'last 5 frames':>18s}")
for tag in ("ref_none", "arc_none", "arc_snap", "arc_hold", "ref_hold"):
    d20, d5 = [], []
    for suffix in ("", "_failed"):
        try:
            f = h5py.File(f"{OUT}/ch_{tag}{suffix}.hdf5", "r")
        except OSError:
            continue
        with f:
            for k in f["data"]:
                pos = f["data"][k]["obs/eef_pos"][:]
                for e in transitions(f["data"][k]["obs/gripper_pos"][:]):
                    if e >= 20:
                        d20.append(np.linalg.norm(pos[e] - pos[e - 20]) * 100)
                        d5.append(np.linalg.norm(pos[e] - pos[e - 5]) * 100)
    d20, d5 = np.array(d20), np.array(d5)
    print(f"  {tag:<10s} {np.median(d20):9.2f} / p90 {np.percentile(d20, 90):6.2f} cm"
          f" {np.median(d5):9.2f} / p90 {np.percentile(d5, 90):5.2f} cm   (n={len(d20)})")
print("  source demos for comparison:", end=" ")
s20, s5 = [], []
with h5py.File(SRC, "r") as f:
    for k in f["data"]:
        pos = f["data"][k]["obs/eef_pos"][:]
        for e in transitions(f["data"][k]["obs/gripper_pos"][:]):
            if e >= 20:
                s20.append(np.linalg.norm(pos[e] - pos[e - 20]) * 100)
                s5.append(np.linalg.norm(pos[e] - pos[e - 5]) * 100)
print(f"{np.median(s20):.2f} cm over 20 frames, {np.median(s5):.2f} cm over 5 (n={len(s20)})")
