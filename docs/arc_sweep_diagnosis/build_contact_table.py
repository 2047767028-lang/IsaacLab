"""Build the reference table for contact_hold_trial.py's snap_ref mode from a finished run.

The four gripper open/close transitions in the recorded gripper_pos mark the four contact events
(close on cube_2, open to release it, close on cube_3, open). The arm's achieved end-effector
position at transition k is what the corrective run holds at subtask k's contact frame. Keyed by
the initial cube layout (env-local frame, rounded to 1e-6) so lookup does not depend on episode
ordering, which differs between asynchronous runs.

usage: build_contact_table.py <run_prefix_without_suffix> <out.npz>
  e.g. build_contact_table.py /path/out/ch_ref_none /path/ref_table.npz
       (reads <prefix>.hdf5 and <prefix>_failed.hdf5)
"""

import sys

import h5py
import numpy as np

CUBES = ("cube_1", "cube_2", "cube_3")
OPEN_VAL = 0.04


def contact_events(grip):
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = jaw < OPEN_VAL - 0.001
    return np.where(np.diff(closed.astype(int)) != 0)[0] + 1


def main(prefix, out):
    keys, poses, kept, skipped = [], [], 0, 0
    for suffix in ("", "_failed"):
        try:
            f = h5py.File(f"{prefix}{suffix}.hdf5", "r")
        except OSError:
            continue
        with f:
            d = f["data"]
            for k in d.keys():
                obs, ro = d[k]["obs"], d[k]["states"]["rigid_object"]
                ev = contact_events(obs["gripper_pos"][:])
                if len(ev) < 4:
                    skipped += 1
                    continue
                eef = obs["eef_pos"][:]
                layout = np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES])
                keys.append(np.round(layout, 6))
                poses.append(np.stack([eef[ev[i]] for i in range(4)]))
                kept += 1
    keys = np.stack(keys)
    poses = np.stack(poses)
    np.savez(out, keys=keys, poses=poses)
    d = np.linalg.norm(np.diff(poses, axis=1), axis=2)
    print(f"kept {kept} episodes, skipped {skipped} without four gripper transitions -> {out}")
    print(f"keys {keys.shape} poses {poses.shape}; median spacing between consecutive contact poses "
          f"{np.round(np.median(d, axis=0) * 100, 2)} cm")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
