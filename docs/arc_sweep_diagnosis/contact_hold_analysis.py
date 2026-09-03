"""Read out the contact-hold runs: success rate, and the mechanism checks that say whether each
intervention actually happened before the success rate is believed.

For every run:
  success        demos in <run>.hdf5 over demos in both files
  grasp reach    gripper-to-cube_2 distance at the first jaw closure (source demos: 0.56 cm median)
  release xy     cube_2-to-cube_1 xy offset at the first jaw opening (tipping threshold 2.34 cm)
  to reference   distance from this run's eef at each of its four contact events to the reference
                 run's eef at the same event in the same scene (needs ref_table.npz)
  length         median episode length, which grows by 4*HOLD if the hold was inserted

usage: contact_hold_analysis.py <out_dir> [tags...]
"""

import os
import sys

import h5py
import numpy as np

CUBES = ("cube_1", "cube_2", "cube_3")
OPEN_VAL = 0.04


def contact_events(grip):
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = jaw < OPEN_VAL - 0.001
    return np.where(np.diff(closed.astype(int)) != 0)[0] + 1


def q(v, scale=100):
    v = np.asarray(v, float) * scale
    if len(v) == 0:
        return "   n/a"
    return f"{np.median(v):5.2f}/{np.percentile(v, 90):5.2f}"


def main(out, tags):
    table = None
    tp = os.path.join(out, "ref_table.npz")
    if os.path.exists(tp):
        blob = np.load(tp)
        table = (blob["keys"], blob["poses"])
    print(f"{'run':<10s} {'success':>12s} {'grasp reach':>12s} {'release xy':>12s} "
          f"{'to ref @1':>10s} {'@2':>10s} {'@3':>10s} {'@4':>10s} {'len':>5s}")
    print("  (cm, median/p90)")
    for tag in tags:
        n_ok = n_all = 0
        reach, relxy, lens = [], [], []
        toref = [[] for _ in range(4)]
        for suffix, ok in (("", True), ("_failed", False)):
            p = os.path.join(out, f"ch_{tag}{suffix}.hdf5")
            if not os.path.exists(p):
                continue
            with h5py.File(p, "r") as f:
                d = f["data"]
                for k in d:
                    n_all += 1
                    n_ok += int(ok)
                    obs, ro = d[k]["obs"], d[k]["states"]["rigid_object"]
                    grip = obs["gripper_pos"][:]
                    eef = obs["eef_pos"][:]
                    c2 = ro["cube_2"]["root_pose"][:, :3]
                    c1 = ro["cube_1"]["root_pose"][:, :3]
                    lens.append(len(eef))
                    ev = contact_events(grip)
                    if len(ev) >= 1:
                        reach.append(np.linalg.norm(eef[ev[0]] - c2[ev[0]]))
                    if len(ev) >= 2:
                        relxy.append(np.linalg.norm((c2[ev[1]] - c1[ev[1]])[:2]))
                    if table is not None and len(ev) >= 4:
                        keys, poses = table
                        layout = np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES])
                        dist = np.linalg.norm(keys - layout, axis=1)
                        i = int(np.argmin(dist))
                        if dist[i] < 2e-3:
                            for j in range(4):
                                toref[j].append(np.linalg.norm(eef[ev[j]] - poses[i, j]))
        if n_all == 0:
            print(f"{tag:<10s}  (no output yet)")
            continue
        rate = f"{n_ok}/{n_all}={100*n_ok/n_all:.1f}%"
        print(f"{tag:<10s} {rate:>12s} {q(reach):>12s} {q(relxy):>12s} "
              + " ".join(f"{q(toref[j]):>10s}" for j in range(4))
              + f" {int(np.median(lens)) if lens else 0:>5d}")


if __name__ == "__main__":
    out = sys.argv[1]
    tags = sys.argv[2:] or ["ref_none", "arc_none", "arc_snap", "arc_hold", "ref_hold"]
    main(out, tags)
