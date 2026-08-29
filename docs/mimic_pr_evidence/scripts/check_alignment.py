"""Are `states` and `obs` recorded at the same instant?

Replaying cubes_stacked from obs/cube_positions reproduces the generator's verdict on 100% of
accepted demos; replaying it from states/rigid_object/*/root_pose reproduces only 98%. If the two
groups are logged on opposite sides of the env step, a one-frame shift should close the gap -- and
which direction it closes in tells us how to line the recorded velocity up with the frame the
criterion actually fired on.
"""

import sys

import h5py
import numpy as np

XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
OPEN_VAL = 0.04
GRIP_TOL = 1e-4 + 1e-4 * OPEN_VAL


def geom(c1, c2, c3):
    out = None
    for a, b in ((c1, c2), (c2, c3)):
        dd = a - b
        ok = np.linalg.norm(dd[:, :2], axis=1) < XY_TH
        ok &= (np.linalg.norm(dd[:, 2:3], axis=1) - H_DIFF < H_TH) & (dd[:, 2] < 0.0)
        out = ok if out is None else (out & ok)
    return out


def main():
    path = sys.argv[1]
    with h5py.File(path, "r") as f:
        d = f["data"]
        keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
        # how far apart are the two position records, frame for frame?
        diffs, results = [], {}
        for shift in (-1, 0, 1):
            results[shift] = 0
        for k in keys:
            st = d[k]["states"]["rigid_object"]
            sp = np.concatenate([st[c]["root_pose"][:, :3] for c in ("cube_1", "cube_2", "cube_3")], axis=1)
            op = d[k]["obs"]["cube_positions"][:]
            diffs.append(float(np.abs(sp - op).max()))

            gp = d[k]["obs"]["gripper_pos"][:]
            jaw = np.maximum(np.abs(gp[:, 0] - OPEN_VAL), np.abs(-gp[:, 1] - OPEN_VAL))
            for shift in (-1, 0, 1):
                s = np.roll(sp, shift, axis=0)
                g = geom(s[:, 0:3], s[:, 3:6], s[:, 6:9])
                if (g & (jaw <= GRIP_TOL)).any():
                    results[shift] += 1

    print(f"{path.split('/')[-1]}: demos={len(keys)}")
    print(f"  max |states.root_pose - obs.cube_positions| across all frames: {max(diffs):.6f} m")
    print("  demos reproducing the accepted verdict, by shift applied to states:")
    for shift, n in results.items():
        print(f"    shift {shift:+d}: {n}/{len(keys)}")


if __name__ == "__main__":
    main()
