"""Did holding the target still actually shrink the residual, or not?

The end-of-subtask dwell was predicted to work from the measured per-step gain of 0.14: a static
target should let the loop close, taking a 1.4 cm residual to a millimetre in about 18 frames. It
did not help -- 20.0% against 22.0% at 3.0 cm. Two very different reasons are possible:

  the dwell did shrink the residual, and the residual was not what was costing the successes; or
  the dwell did not shrink the residual, and the model of "static target lets the loop converge"
  is simply wrong.

Both runs at a given intervention share a seed and a scene sequence, so pairing the 3.0 cm run
against the 0.5 cm one measures the residual directly, with and without the dwell.
"""

import h5py
import numpy as np

OUT = "/home/pk/.claude/jobs/10fee75c/tmp/out"
CUBES = ("cube_1", "cube_2", "cube_3")


def key_of(ro):
    return tuple(np.round(np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]), 6))


def load(tag):
    out = {}
    for suffix, ok in (("", True), ("_failed", False)):
        try:
            f = h5py.File(f"{OUT}/fix_{tag}{suffix}.hdf5", "r")
        except OSError:
            continue
        with f:
            d = f["data"]
            for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
                out[key_of(d[k]["states"]["rigid_object"])] = {
                    "success": ok,
                    "pos": d[k]["obs"]["eef_pos"][:],
                    "act": d[k]["obs"]["actions"][:, :3],
                }
    return out


def compare(low_tag, high_tag, label):
    lo, hi = load(low_tag), load(high_tag)
    devs, lag_lo, lag_hi = [], [], []
    for k, a in hi.items():
        r = lo.get(k)
        if r is None or not (a["success"] and r["success"]):
            continue
        n = min(15, len(a["pos"]), len(r["pos"]))
        devs.append(float(np.linalg.norm(a["pos"][-n:] - r["pos"][-n:], axis=1).max()))
        # tracking lag over the same window, to see whether the hold let either arm catch up
        lag_lo.append(float(np.median(np.linalg.norm(r["act"][-n:], axis=1))))
        lag_hi.append(float(np.median(np.linalg.norm(a["act"][-n:], axis=1))))
    devs = np.array(devs)
    print(
        f"  {label:<24s} paired n={len(devs):4d}   residual median={np.median(devs) * 100:6.3f}cm"
        f"   p90={np.percentile(devs, 90) * 100:6.3f}cm"
        f"   |action| in window: low={np.median(lag_lo) * 100:5.2f}cm high={np.median(lag_hi) * 100:5.2f}cm"
    )


def main():
    print("paired 3.0 cm against 0.5 cm, scenes succeeding at both, last 15 frames\n")
    compare("ref_low", "ref_high", "no intervention")
    compare("dwell_low", "dwell_high", "dwell at subtask start")
    compare("tail_low", "tail_high", "dwell at subtask end")
    compare("scale_low", "scale_high", "controller gain 1.0")
    print("\n  a dwell that works shows a smaller residual than 'no intervention'.")


if __name__ == "__main__":
    main()
