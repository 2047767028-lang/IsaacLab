"""How far does the source arm move in the 10-20 frames after a contact event?

The previous direction-2 patch aimed the subtask's LAST target frame (= contact + 10..20 frames)
at the reference's pose AT contact. If the arm moves several cm in those frames, that delta was a
retarget in the wrong direction, independent of any lag argument.
"""
import h5py
import numpy as np

F = "/home/pk/IsaacLab/datasets/annotated_dataset.hdf5"
OPEN_VAL = 0.04


def events(grip):
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = jaw < OPEN_VAL - 0.001
    return np.where(np.diff(closed.astype(int)) != 0)[0] + 1


rows = {10: [], 15: [], 20: []}
with h5py.File(F, "r") as f:
    for k in f["data"]:
        pos = f["data"][k]["obs/eef_pos"][:]
        for e in events(f["data"][k]["obs/gripper_pos"][:]):
            for n in rows:
                if e + n < len(pos):
                    rows[n].append(np.linalg.norm(pos[e + n] - pos[e]) * 100)
for n, v in rows.items():
    v = np.array(v)
    print(f"|eef(contact+{n}) - eef(contact)|  median {np.median(v):5.2f} cm  p90 {np.percentile(v, 90):5.2f} cm  (n={len(v)})")
