"""Does angular velocity separate the two defects that linear velocity alone lets through?

The linear-speed distributions overlap slightly: intact demos top out at 0.0341 m/s and the two
slowest defective ones sit at 0.0278 and 0.0283, so no linear threshold is a perfect separator. A
cube caught near the apex of a bounce is briefly slow in translation but still tumbling, while a
cube resting on another is neither, so angular velocity is worth checking before settling for a
partial fix.
"""

import h5py
import numpy as np

from threshold_v2 import CUBES, GRIP_TOL, OPEN_VAL, geom

PATH = "/home/pk/.claude/jobs/10fee75c/tmp/out/C.hdf5"


def main():
    rows = []
    with h5py.File(PATH, "r") as f:
        d = f["data"]
        for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
            st = d[k]["states"]
            ro = st["rigid_object"]
            poses = [ro[c]["root_pose"][:] for c in CUBES]
            lin = np.stack([np.linalg.norm(ro[c]["root_velocity"][:, :3], axis=1) for c in CUBES]).max(0)
            ang = np.stack([np.linalg.norm(ro[c]["root_velocity"][:, 3:6], axis=1) for c in CUBES]).max(0)
            jp = st["articulation"]["robot"]["joint_position"][:]
            jaw = np.maximum(np.abs(jp[:, 7] - OPEN_VAL), np.abs(jp[:, 8] - OPEN_VAL))
            g = geom(*poses)
            rows.append({"k": k, "g": g, "q": g & (jaw <= GRIP_TOL), "lin": lin, "ang": ang})

    broken = np.array([not bool(r["g"][-1]) for r in rows])
    # Evaluate at the qualifying frame the cube was most settled on -- the frame a threshold would
    # have to reject to reject the demo.
    lins = np.array([r["lin"][r["q"]].min() for r in rows])
    angs = np.array([r["ang"][r["q"]][np.argmin(r["lin"][r["q"]])] for r in rows])

    for label, m in (("intact", ~broken), ("broken", broken)):
        print(
            f"{label:7s} n={m.sum():3d}  angular p50={np.median(angs[m]):.4f}"
            f"  p90={np.percentile(angs[m], 90):.4f}  max={angs[m].max():.4f} rad/s"
        )

    print("\ncombined (linear < 0.05 m/s and angular < t):")
    for t in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
        p = (lins < 0.05) & (angs < t)
        print(f"  angular < {t:4.1f}:  intact {p[~broken].sum():2d}/{(~broken).sum()}   broken {p[broken].sum()}/{broken.sum()}")


if __name__ == "__main__":
    main()
