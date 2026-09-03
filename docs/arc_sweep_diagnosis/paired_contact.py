"""Paired test: does the arc actually leave the contact phase alone?

The scene sequence is identical across amplitudes within a seed (verified: all 500 initial cube
layouts match to <1e-6 m), so attempt k at 0.5 cm and attempt k at 3.0 cm start from exactly the
same placements. That removes scene difficulty from the comparison entirely.

The claim under test is that because the envelope is zero in value and slope at both ends of the
free zone, and the trailing 30% of every subtask is byte-identical to the source, the contact phase
is unaffected. That is true of the TARGET pose sequence. This measures the ACHIEVED one.

Two paired quantities, both restricted to scenes that SUCCEED at both amplitudes, so neither is
contaminated by which episodes failed:

  1. end-of-episode end-effector deviation between the paired runs -- the last frames sit inside the
     final subtask's frozen tail, where the two runs are commanded identically. Any deviation there
     is tracking error carried in from the perturbed free space.
  2. placement error, cube by cube, as a paired difference.

A third block reports the outcome flips, which is where the lost success rate actually comes from.
"""

import sys

import h5py
import numpy as np

ROOT = "/home/pk/IsaacLab/datasets/perturbation_sweep_v2_arc"
XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
CUBES = ("cube_1", "cube_2", "cube_3")
TAIL = 15
"""Frames from the end to compare. The final subtask's frozen tail is ~30% of its length, and
subtasks run tens of frames, so this window sits inside it."""


def key_of(ro):
    return tuple(np.round(np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]), 6))


def load_run(seed, cm):
    """Return {layout_key: episode dict} across both the success and failure files."""
    tag = f"cm{cm:.1f}".replace(".", "p")
    out = {}
    for suffix, ok in (("", True), ("_failed", False)):
        with h5py.File(f"{ROOT}/seed{seed}/generated_{tag}{suffix}.hdf5", "r") as f:
            d = f["data"]
            for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
                ro = d[k]["states"]["rigid_object"]
                pos = [ro[c]["root_pose"][:, :3] for c in CUBES]
                out[key_of(ro)] = {
                    "success": ok,
                    "eef": d[k]["obs"]["eef_pos"][:],
                    "final": [p[-1] for p in pos],
                    "T": pos[0].shape[0],
                }
    return out


def xy(a, b):
    return float(np.linalg.norm((a - b)[:2]))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ref_cm = 0.5
    ref = load_run(seed, ref_cm)
    print(f"seed={seed}   reference amplitude {ref_cm} cm, {len(ref)} attempts\n")

    print("=== paired outcome flips (same scene, both amplitudes) ===")
    print(f"  {'cm':>5s} {'paired':>7s} {'both ok':>8s} {'ref only':>9s} {'other only':>11s} {'neither':>8s}")
    flips = {}
    for cm in (0.6, 1.0, 1.5, 2.0, 2.5, 3.0):
        other = load_run(seed, cm)
        common = [k for k in ref if k in other]
        a = np.array([ref[k]["success"] for k in common])
        b = np.array([other[k]["success"] for k in common])
        flips[cm] = (common, other, a, b)
        print(
            f"  {cm:5.1f} {len(common):7d} {(a & b).sum():8d} {(a & ~b).sum():9d}"
            f" {(~a & b).sum():11d} {(~a & ~b).sum():8d}"
        )

    print("\n=== achieved end-effector deviation over the last "
          f"{TAIL} frames, scenes succeeding at BOTH ===")
    print(f"  {'cm':>5s} {'n':>5s} {'median':>9s} {'p90':>9s} {'max':>9s}")
    for cm, (common, other, a, b) in flips.items():
        devs = []
        for k in common:
            if not (ref[k]["success"] and other[k]["success"]):
                continue
            e1, e2 = ref[k]["eef"], other[k]["eef"]
            n = min(TAIL, len(e1), len(e2))
            devs.append(float(np.linalg.norm(e1[-n:] - e2[-n:], axis=1).max()))
        devs = np.array(devs)
        print(
            f"  {cm:5.1f} {len(devs):5d} {np.median(devs) * 100:8.3f}cm"
            f" {np.percentile(devs, 90) * 100:8.3f}cm {devs.max() * 100:8.3f}cm"
        )

    print("\n=== paired placement error, scenes succeeding at BOTH (other minus reference) ===")
    print(f"  {'cm':>5s} {'n':>5s} {'d xy(c1,c2)':>12s} {'d xy(c2,c3)':>12s} {'worse on c1-c2':>15s}")
    for cm, (common, other, a, b) in flips.items():
        d12, d23 = [], []
        for k in common:
            if not (ref[k]["success"] and other[k]["success"]):
                continue
            r, o = ref[k]["final"], other[k]["final"]
            d12.append(xy(o[0], o[1]) - xy(r[0], r[1]))
            d23.append(xy(o[1], o[2]) - xy(r[1], r[2]))
        d12, d23 = np.array(d12), np.array(d23)
        print(
            f"  {cm:5.1f} {len(d12):5d} {np.median(d12) * 100:+11.4f}cm {np.median(d23) * 100:+11.4f}cm"
            f" {(d12 > 0).mean() * 100:14.1f}%"
        )


if __name__ == "__main__":
    main()
