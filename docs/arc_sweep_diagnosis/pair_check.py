"""Can episodes be paired across amplitudes within one seed?

If the sweep's per-run seed produces the same sequence of scene layouts regardless of arc amplitude,
then attempt k at 0.5 cm and attempt k at 3.0 cm start from the same cube placements, and the two
runs can be compared attempt by attempt. That would make it possible to measure directly what the
arc does to the achieved trajectory rather than inferring it from aggregate rates.

Drawing the arc direction consumes RNG, so the sequences may well have diverged; this establishes
which it is before anything is built on top.

The generated files hold only the successes and the failures separately, and each is a filtered
subsequence of the attempt stream, so the comparison is over the multiset of initial cube layouts
from both files together.
"""

import h5py
import numpy as np

ROOT = "/home/pk/IsaacLab/datasets/perturbation_sweep_v2_arc"
CUBES = ("cube_1", "cube_2", "cube_3")


def initial_layouts(seed, cm):
    tag = f"cm{cm:.1f}".replace(".", "p")
    rows = []
    for suffix in ("", "_failed"):
        p = f"{ROOT}/seed{seed}/generated_{tag}{suffix}.hdf5"
        with h5py.File(p, "r") as f:
            d = f["data"]
            for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
                ro = d[k]["states"]["rigid_object"]
                rows.append(np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]))
    return np.array(rows)


def main():
    seed = 1
    base = initial_layouts(seed, 0.5)
    print(f"seed={seed}  0.5cm layouts: {base.shape}")
    for cm in (0.6, 1.0, 3.0):
        other = initial_layouts(seed, cm)
        # nearest-neighbour distance from each layout at `cm` to the closest 0.5 cm layout
        d = np.linalg.norm(other[:, None, :] - base[None, :, :], axis=2)
        nn = d.min(axis=1)
        exact = (nn < 1e-6).sum()
        print(
            f"  {cm:.1f}cm: n={len(other)}  layouts also present at 0.5cm (<1e-6): {exact}"
            f" ({exact / len(other) * 100:.1f}%)   median NN distance={np.median(nn):.4f} m"
        )


if __name__ == "__main__":
    main()
