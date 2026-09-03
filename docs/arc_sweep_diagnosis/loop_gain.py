"""How much of a commanded correction does the arm actually execute in one step?

The action returned by `target_eef_pose_to_action` is `target_pos - curr_pos` in metres, and the
DifferentialInverseKinematicsActionCfg applies `scale=0.5`, so the controller is commanded to close
half the remaining gap each step. If the arm executed that, a displacement injected in free space
would decay as 0.5^n and be gone within a couple of frames of the frozen tail, which runs about 17
frames. The measured residual instead survives the whole tail.

So the question is the effective per-step gain: achieved displacement divided by the position error
the action encodes. That number sets the decay time constant of any injected displacement, and with
it whether a frozen tail of any practical length could absorb one.

Measured on the unperturbed-ish reference run (0.5 cm), where nothing is being injected, so it
characterises the controller rather than the perturbation.
"""

import sys

import h5py
import numpy as np

ROOT = "/home/pk/IsaacLab/datasets/perturbation_sweep_v2_arc"
SCALE = 0.5
"""DifferentialInverseKinematicsActionCfg(scale=0.5) in stack_ik_rel_env_cfg.py."""


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cm = sys.argv[2] if len(sys.argv) > 2 else "0p5"
    p = f"{ROOT}/seed{seed}/generated_cm{cm}.hdf5"

    err_norm, ach_norm, ratio, cos = [], [], [], []
    with h5py.File(p, "r") as f:
        d = f["data"]
        for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1]))[:60]:
            act = d[k]["obs"]["actions"][:, :3]  # target - current, metres
            eef = d[k]["obs"]["eef_pos"][:]
            ach = np.diff(eef, axis=0)  # achieved displacement over the step
            e = act[:-1]
            ne, na = np.linalg.norm(e, axis=1), np.linalg.norm(ach, axis=1)
            m = ne > 1e-4  # ignore steps where there is essentially nothing to correct
            err_norm.append(ne[m])
            ach_norm.append(na[m])
            ratio.append(na[m] / ne[m])
            cos.append(np.sum(ach[m] * e[m], axis=1) / (na[m] * ne[m] + 1e-12))

    err_norm = np.concatenate(err_norm)
    ach_norm = np.concatenate(ach_norm)
    ratio = np.concatenate(ratio)
    cos = np.concatenate(cos)

    print(f"seed={seed} file=cm{cm}  steps={len(ratio)}\n")
    print(f"  position error carried by the action : median {np.median(err_norm) * 100:.3f} cm")
    print(f"  achieved displacement per step       : median {np.median(ach_norm) * 100:.3f} cm")
    print(f"  commanded (scale {SCALE})              : median {np.median(err_norm) * SCALE * 100:.3f} cm")
    print(f"\n  effective per-step gain (achieved / error): median {np.median(ratio):.4f}"
          f"   p10={np.percentile(ratio, 10):.4f}  p90={np.percentile(ratio, 90):.4f}")
    print(f"  fraction of the commanded delta executed  : median {np.median(ratio) / SCALE:.4f}")
    print(f"  alignment of achieved with error direction: median cos={np.median(cos):.3f}")

    g = float(np.median(ratio))
    if 0 < g < 1:
        half = np.log(0.5) / np.log(1 - g)
        print(f"\n  decay of an injected displacement at gain {g:.4f}:")
        print(f"    half-life {half:.1f} frames;  after a 17-frame frozen tail, "
              f"{(1 - g) ** 17 * 100:.1f}% remains")
        for tail in (17, 30, 60):
            print(f"    tail of {tail:3d} frames -> {(1 - g) ** tail * 100:5.1f}% of the injection survives")


if __name__ == "__main__":
    main()
