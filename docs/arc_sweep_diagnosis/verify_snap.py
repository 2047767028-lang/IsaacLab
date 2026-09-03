"""Did the correction actually put the gripper where the reference arm was?

Direction 2 came back worse than its control -- 12.0% and 13.0% against 15.7% -- but that only
refutes the idea if the gripper really did end up where the reference's gripper was. If the target
moved and the arm never arrived, nothing has been tested and the physics argument stands.

The reference table holds the reference run's achieved pose at each of the four contact events, so
the distance from the corrected run's own contact poses to those is measured directly. Episodes are
matched by initial cube layout, which reseeding makes reproducible to about a millimetre.
"""

import h5py
import numpy as np

OUT = "/home/pk/.claude/jobs/10fee75c/tmp/out"
CUBES = ("cube_1", "cube_2", "cube_3")
OPEN_VAL = 0.04


def contact_events(grip):
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = jaw < OPEN_VAL - 0.001
    return np.where(np.diff(closed.astype(int)) != 0)[0] + 1


def scan(tag, keys, poses):
    per_event = [[] for _ in range(4)]
    matched = total = 0
    for suffix in ("", "_failed"):
        try:
            f = h5py.File(f"{OUT}/fix_{tag}{suffix}.hdf5", "r")
        except OSError:
            continue
        with f:
            d = f["data"]
            for k in d:
                obs, ro = d[k]["obs"], d[k]["states"]["rigid_object"]
                total += 1
                layout = np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES])
                dist = np.linalg.norm(keys - layout, axis=1)
                i = int(np.argmin(dist))
                if dist[i] >= 2e-3:
                    continue
                ev = contact_events(obs["gripper_pos"][:])
                if len(ev) < 4:
                    continue
                matched += 1
                eef = obs["eef_pos"][:]
                for j in range(4):
                    per_event[j].append(float(np.linalg.norm(eef[ev[j]] - poses[i, j])))
    med = [np.median(v) * 100 if v else float("nan") for v in per_event]
    print(
        f"  {tag:14s} episodes={total:4d} matched={matched:4d}   "
        "distance to the reference's contact pose: " + " ".join(f"{m:6.2f}cm" for m in med)
    )


def main():
    blob = np.load("/home/pk/.claude/jobs/10fee75c/tmp/ref_poses.npz")
    keys, poses = blob["keys"], blob["poses"]
    print("how far each run's gripper was from the reference run's, at the four contact events\n")
    scan("d2b_arc", keys, poses)
    scan("f_snap_only", keys, poses)
    scan("f_snap_hold", keys, poses)
    scan("f_hold_only", keys, poses)
    print("\n  d2b_arc is the uncorrected control; if the correction worked, the rows below it")
    print("  should show smaller distances. If they do not, the target moved but the arm did not.")


if __name__ == "__main__":
    main()
