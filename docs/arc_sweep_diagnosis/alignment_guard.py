"""Is the paired tail deviation real, or an artefact of misaligned episode lengths?

paired_contact.py compares the last N frames of two runs by index from the end. The arm moves at
roughly 4 mm per frame, so a three-frame length difference would manufacture about 1.2 cm of
apparent deviation -- the same magnitude as the result. If episode lengths differ between paired
runs, that measurement is worthless.

Three checks:
  1. do paired episodes have equal length?
  2. if lengths are equal, is the deviation still there when compared frame-for-frame from the start
     as well as from the end? (Equal length makes both alignments the same, so this is a tautology
     check that the pairing is what it claims to be.)
  3. the best-case reading: minimum deviation over any time shift in [-5, +5]. If a small shift
     collapses the deviation, the effect is timing; if it does not, the arm is genuinely elsewhere.
"""

import sys

import h5py
import numpy as np

ROOT = "/home/pk/IsaacLab/datasets/perturbation_sweep_v2_arc"
CUBES = ("cube_1", "cube_2", "cube_3")
TAIL = 15


def key_of(ro):
    return tuple(np.round(np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]), 6))


def load_run(seed, cm):
    tag = f"cm{cm:.1f}".replace(".", "p")
    out = {}
    for suffix, ok in (("", True), ("_failed", False)):
        with h5py.File(f"{ROOT}/seed{seed}/generated_{tag}{suffix}.hdf5", "r") as f:
            d = f["data"]
            for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
                eef = d[k]["obs"]["eef_pos"][:]
                out[key_of(d[k]["states"]["rigid_object"])] = {"success": ok, "eef": eef, "T": len(eef)}
    return out


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ref = load_run(seed, 0.5)

    print(f"seed={seed}, paired against 0.5 cm, scenes succeeding at both\n")
    print(f"  {'cm':>5s} {'n':>5s} {'equal length':>13s} {'median |dT|':>12s} {'tail dev':>10s}"
          f" {'best-shift dev':>15s} {'shift used (med)':>17s} {'eef speed/frame':>16s}")
    for cm in (0.6, 1.0, 2.0, 3.0):
        other = load_run(seed, cm)
        dT, dev, best, shifts, speeds = [], [], [], [], []
        for k in ref:
            if k not in other or not (ref[k]["success"] and other[k]["success"]):
                continue
            e1, e2 = ref[k]["eef"], other[k]["eef"]
            dT.append(len(e2) - len(e1))
            n = min(TAIL, len(e1), len(e2))
            dev.append(float(np.linalg.norm(e1[-n:] - e2[-n:], axis=1).max()))
            speeds.append(float(np.median(np.linalg.norm(np.diff(e1[-n:], axis=0), axis=1))))
            # best alignment over a small shift of the second run
            cand = []
            for s in range(-5, 6):
                a_end, b_end = len(e1), len(e2) - s
                if b_end - n < 0 or b_end > len(e2) or a_end - n < 0:
                    continue
                cand.append((float(np.linalg.norm(e1[a_end - n : a_end] - e2[b_end - n : b_end], axis=1).max()), s))
            if cand:
                v, s = min(cand)
                best.append(v)
                shifts.append(s)
        dT = np.array(dT)
        print(
            f"  {cm:5.1f} {len(dT):5d} {(dT == 0).mean() * 100:12.1f}% {np.median(np.abs(dT)):12.1f}"
            f" {np.median(dev) * 100:9.3f}cm {np.median(best) * 100:14.3f}cm {np.median(shifts):17.1f}"
            f" {np.median(speeds) * 100:15.3f}cm"
        )

    print("\n  reading: if 'equal length' is 100% the end-alignment is exact and the tail deviation")
    print("  stands; if 'best-shift dev' is far below 'tail dev', the effect was timing, not position.")


if __name__ == "__main__":
    main()
