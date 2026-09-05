"""Per-scene paired analysis across the v2 policy evaluations.

All five evals ran the same 300 scenes (--seed 101 fixes the reproducible scene sequence), so the
per-trial records can be aligned by trial index. Previous analysis only ever used the pooled
success rate, which throws that alignment away.

Three questions this answers, none of which the pooled numbers can:

1. How much of the outcome is decided by the scene? Very little. The permutation test says the
   scene effect is statistically real (p=1e-4), but agreement between any two models sits only
   -1 to +6 pp above what independent coin flips would give.
2. How much do two models trained on identical data, differing only in training seed, disagree at
   the level of individual scenes? 40-45% of scenes flip, while the net success rate moves only
   ~3.5pp -- the flips very nearly cancel. That is the microscopic mechanism behind the +-4pp
   training-seed noise floor.
3. Does sharing --seed 101 across models buy the paired-comparison variance reduction it was
   assumed to? No. phi is -0.02 to +0.13, so McNemar gains essentially nothing over an unpaired
   test here. Sharing the seed is still the right call; just do not count on it for power.

The mechanism behind 1 and 2 is in openpi's policy.py: pi0.5 sampling is stochastic and the RNG
state advances on every infer() call, so which noise draw a given rollout gets depends on the total
step count of every rollout before it. That does not threaten the pooled success rates -- each
rollout is still an independent Bernoulli trial and the Wilson interval covers it -- but it does
mean per-scene outcomes carry an effectively arbitrary draw.

Usage:  python paired_scene_analysis.py [results_dir]
"""

import glob
import itertools
import json
import sys

import numpy as np


def load(results_dir: str):
    names, mat = [], []
    for path in sorted(glob.glob(f"{results_dir}/*.json")):
        data = json.load(open(path))
        records = data["records"]
        # Alignment is the whole point -- refuse to proceed if trial indices are not 0..n-1 in order.
        assert [r["trial"] for r in records] == list(range(len(records))), f"{path}: trials not in order"
        names.append(path.split("/")[-1].replace(".json", ""))
        mat.append(np.array([r["success"] for r in records], dtype=int))
    lengths = {len(m) for m in mat}
    assert len(lengths) == 1, f"evals have differing trial counts: {lengths}"
    return names, np.stack(mat)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    names, s = load(results_dir)
    p = s.mean(1)
    print(f"models: {names}\nshape: {s.shape}\n")
    for n, rate in zip(names, p):
        print(f"  {n:16s} {rate * 100:5.2f}%")

    print("\n=== agreement vs. what independence would give ===")
    for a, b in itertools.combinations(range(len(names)), 2):
        obs = (s[a] == s[b]).mean()
        exp = p[a] * p[b] + (1 - p[a]) * (1 - p[b])
        phi = np.corrcoef(s[a], s[b])[0, 1]
        print(
            f"  {names[a]:16s} vs {names[b]:16s} "
            f"observed={obs * 100:5.1f}%  independent={exp * 100:5.1f}%  "
            f"excess={(obs - exp) * 100:+5.1f}pp  phi={phi:+.3f}"
        )

    k = s.sum(0)
    print("\n=== how many models solved each scene ===")
    for i in range(len(names) + 1):
        print(f"  {i}/{len(names)}: {(k == i).sum():3d} scenes ({(k == i).mean() * 100:5.1f}%)")

    # Permutation test: if outcomes were independent across models, k would be Poisson-binomial and
    # its variance would be sum p(1-p). Excess variance means some scenes really are harder.
    rng = np.random.default_rng(0)
    obs_var = k.var()
    null = np.array([np.stack([rng.permutation(row) for row in s]).sum(0).var() for _ in range(20000)])
    print("\n=== permutation test for a scene effect (20000 draws) ===")
    print(f"  observed var(k) = {obs_var:.4f}")
    print(f"  null mean = {null.mean():.4f}  95% = [{np.percentile(null, 2.5):.4f}, {np.percentile(null, 97.5):.4f}]")
    print(f"  empirical p = {(null >= obs_var).mean():.4f}")

    print("\n=== same dataset, different training seed ===")
    for a, b in itertools.combinations(range(len(names)), 2):
        # Pair up runs whose names differ only by the seed suffix.
        if names[a].rsplit("_s", 1)[0] != names[b].rsplit("_s", 1)[0]:
            continue
        flips = (s[a] != s[b]).sum()
        a_only = int(((s[a] == 1) & (s[b] == 0)).sum())
        b_only = int(((s[a] == 0) & (s[b] == 1)).sum())
        n = s.shape[1]
        print(
            f"  {names[a]:16s} vs {names[b]:16s} "
            f"scene-level disagreement {flips:3d}/{n} = {flips / n * 100:.1f}%  "
            f"(A only {a_only} / B only {b_only})  net {(a_only - b_only) / n * 100:+.2f}pp"
        )


if __name__ == "__main__":
    main()
