"""During a dwell, does the arm actually catch up to its target?

The pairing measurement came out with two to eleven matched scenes -- the ten asynchronous
environments do not reset in a reproducible order in this harness, unlike the sweep's files -- so it
cannot answer anything. This is a within-run measure instead, and every episode contributes.

The action is `target - current`, so while the target is held still it should decay geometrically
and the arm should end up close to its target. If a dwell works, episodes should contain frames
where that quantity is small. If it does not, the arm stays about as far behind as ever, and the
prediction that a static target lets the loop close was wrong.

The injected action noise (sigma 3 cm per axis) is off during the fixed segment
(`apply_noise_during_interpolation=False`), so those frames are the clean ones.
"""

import h5py
import numpy as np

OUT = "/home/pk/.claude/jobs/10fee75c/tmp/out"


def stats(tag):
    with h5py.File(f"{OUT}/fix_{tag}.hdf5", "r") as f:
        d = f["data"]
        keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
        mins, p05, frac_small, lens = [], [], [], []
        for k in keys:
            a = np.linalg.norm(d[k]["obs"]["actions"][:, :3], axis=1)
            mins.append(a.min())
            p05.append(np.percentile(a, 5))
            frac_small.append(float((a < 0.01).mean()))
            lens.append(len(a))
    return keys, np.array(mins), np.array(p05), np.array(frac_small), np.array(lens)


def main():
    print("|target - current| within each episode; the dwell frames are the ones with no noise\n")
    print(f"  {'run':<14s} {'n':>4s} {'len':>6s} {'min':>9s} {'p05':>9s} {'frames < 1 cm':>14s}")
    for tag in ("ref_high", "dwell_high", "tail_high", "tail2_high", "ref_low", "dwell_low", "tail_low", "tail2_low"):
        try:
            keys, mins, p05, frac, lens = stats(tag)
        except OSError:
            print(f"  {tag:<14s} (missing)")
            continue
        print(
            f"  {tag:<14s} {len(keys):4d} {np.median(lens):6.0f} {np.median(mins) * 100:8.3f}cm"
            f" {np.median(p05) * 100:8.3f}cm {np.median(frac) * 100:13.1f}%"
        )

    print("\n  a working dwell shows a much smaller minimum and a visible population of near-zero")
    print("  frames; if these columns barely move, holding the target still did not let the arm")
    print("  close the gap, whatever the per-step gain suggested.")


if __name__ == "__main__":
    main()
