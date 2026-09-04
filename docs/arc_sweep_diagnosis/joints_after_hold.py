"""After the contact-frame hold has converged the end-effector, does the joint configuration still
carry any of the remaining arc penalty?

Pairs arc_hold with ref_hold by initial layout (same reseeded scenes). At the first jaw closure:
  joint diff      |q_arc - q_ref| over the 7 arm joints (rad)
  eef pos diff    |p_arc - p_ref| (cm)
  eef rot diff    angle between the two eef quaternions (deg)
  cube nudge      how far cube_2 moved between the episode start and the grasp frame (cm) -- was it
                  pushed during the approach?
  seat xy         eef-to-cube_2 xy offset at closure (cm) -- where the cube sits between the fingers
And the outcome: did cube_2 end up on cube_1 (placement)?

Each quantity is split by the arc run's placement outcome, with Cohen's d. A quantity that is the
carrier separates the two groups; a proxy collapses once the eef pose is matched.
"""
import h5py
import numpy as np

OUT = "/home/pk/IsaacLab/datasets/arc_sweep_diagnosis_runs/contact_hold"
CUBES = ("cube_1", "cube_2", "cube_3")
OPEN_VAL = 0.04


def transitions(grip):
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = jaw < OPEN_VAL - 0.001
    return np.where(np.diff(closed.astype(int)) != 0)[0] + 1


def quat_angle(q1, q2):  # w,x,y,z
    d = abs(float(np.dot(q1, q2)))
    return np.degrees(2 * np.arccos(min(1.0, d)))


def load(tag):
    eps = []
    for suffix, ok in (("", True), ("_failed", False)):
        with h5py.File(f"{OUT}/ch_{tag}{suffix}.hdf5", "r") as f:
            for k in f["data"]:
                d = f["data"][k]
                ro = d["states"]["rigid_object"]
                eps.append({
                    "key": np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]),
                    "ok": ok,
                    "q": d["obs/joint_pos"][:, :7],
                    "p": d["obs/eef_pos"][:],
                    "quat": d["obs/eef_quat"][:],
                    "grip": d["obs/gripper_pos"][:],
                    "c1": ro["cube_1"]["root_pose"][:, :3],
                    "c2": ro["cube_2"]["root_pose"][:, :3],
                })
    return eps


def d_of(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((a.var() + b.var()) / 2) or 1e-12
    return (b.mean() - a.mean()) / pooled


ref = load("ref_hold")
arc = load("arc_hold")
ref_keys = np.stack([e["key"] for e in ref])
rows = []
for a in arc:
    dist = np.linalg.norm(ref_keys - a["key"], axis=1)
    i = int(np.argmin(dist))
    if dist[i] > 2e-3:
        continue
    r = ref[i]
    ta, tr = transitions(a["grip"]), transitions(r["grip"])
    if len(ta) < 2 or len(tr) < 2:
        continue
    ga, gr = ta[0], tr[0]
    rows.append({
        "placed": np.linalg.norm((a["c2"][-1] - a["c1"][-1])[:2]) < 0.02 and (a["c2"][-1, 2] - a["c1"][-1, 2]) > 0.03,
        "joint": np.linalg.norm(a["q"][ga] - r["q"][gr]),
        "pos": np.linalg.norm(a["p"][ga] - r["p"][gr]) * 100,
        "rot": quat_angle(a["quat"][ga], r["quat"][gr]),
        "nudge": np.linalg.norm(a["c2"][ga] - a["c2"][0]) * 100,
        "nudge_ref": np.linalg.norm(r["c2"][gr] - r["c2"][0]) * 100,
        "seat": np.linalg.norm((a["p"][ga] - a["c2"][ga])[:2]) * 100,
        "seat_ref": np.linalg.norm((r["p"][gr] - r["c2"][gr])[:2]) * 100,
        "rel_xy": np.linalg.norm((a["c2"][ta[1]] - a["c1"][ta[1]])[:2]) * 100,
    })
placed = np.array([x["placed"] for x in rows])
print(f"paired arc_hold vs ref_hold episodes: {len(rows)}   arc placed cube_2: {placed.sum()} ({100*placed.mean():.1f}%)")
print(f"  {'quantity at first grasp (arc run)':<40s} {'placed':>14s} {'not placed':>14s} {'d':>6s}")
for name, key in (
    ("joint config vs ref (rad)", "joint"),
    ("eef position vs ref (cm)", "pos"),
    ("eef orientation vs ref (deg)", "rot"),
    ("cube_2 nudged before grasp, arc (cm)", "nudge"),
    ("cube_2 nudged before grasp, ref (cm)", "nudge_ref"),
    ("seat: eef-cube_2 xy at closure, arc (cm)", "seat"),
    ("seat: eef-cube_2 xy at closure, ref (cm)", "seat_ref"),
    ("cube_2-cube_1 xy at release, arc (cm)", "rel_xy"),
):
    v = np.array([x[key] for x in rows])
    print(f"  {name:<40s} {np.median(v[placed]):6.3f}/{np.percentile(v[placed], 90):6.3f} "
          f"{np.median(v[~placed]):6.3f}/{np.percentile(v[~placed], 90):6.3f} {d_of(v[placed], v[~placed]):+6.2f}")
print("  (median/p90; d>0 means larger in the failed placements)")

# does the joint difference predict anything once seat is controlled? crude: within the narrow-seat band
seat = np.array([x["seat"] for x in rows]); joint = np.array([x["joint"] for x in rows])
band = seat < np.median(seat)
print(f"\n  within the better-seated half (seat < {np.median(seat):.2f} cm): joint-diff d = "
      f"{d_of(joint[band & placed], joint[band & ~placed]):+.2f}   (n placed {int((band & placed).sum())}, failed {int((band & ~placed).sum())})")
band = ~band
print(f"  within the worse-seated half:                      joint-diff d = "
      f"{d_of(joint[band & placed], joint[band & ~placed]):+.2f}   (n placed {int((band & placed).sum())}, failed {int((band & ~placed).sum())})")
