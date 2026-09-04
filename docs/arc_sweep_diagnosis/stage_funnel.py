"""Stage funnel for any set of contact-hold runs: fraction of attempts that lift cube_2, place it on
cube_1, then lift cube_3, and finally pass the task criterion.

usage: stage_funnel.py <out_dir> <tag> [<tag> ...]
"""
import os
import sys

import h5py
import numpy as np


def main(out, tags):
    print(f"  {'run':<10s} {'cube_2 lifted':>13s} {'on cube_1':>10s} {'cube_3 lifted':>13s} {'success':>8s}")
    for tag in tags:
        n = g1 = p1 = g2 = ok = 0
        for suffix, succ in (("", True), ("_failed", False)):
            p = os.path.join(out, f"ch_{tag}{suffix}.hdf5")
            if not os.path.exists(p):
                continue
            with h5py.File(p, "r") as f:
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
        if n == 0:
            print(f"  {tag:<10s}  (no output)")
            continue
        print(f"  {tag:<10s} {100*g1/n:12.1f}% {100*p1/n:9.1f}% {100*g2/n:12.1f}% {100*ok/n:7.1f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
