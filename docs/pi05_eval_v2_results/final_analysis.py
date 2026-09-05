"""Final read-out: three datasets, plus a run-to-run noise floor to judge them against.

The whole point of the seed-777 pair is that a difference between datasets only means something if
it is bigger than the difference you get by retraining the SAME dataset with a different seed. v1
went wrong for exactly the lack of that yardstick.
"""
import json
import math
import os
import sys

RES = "/home/pk/.claude/jobs/2cbde0b3/tmp/final"

RUNS = {
    "baseline_s42": "baseline (380 干净), 种子42",
    "baseline_s777": "baseline (380 干净), 种子777",
    "arc_s42": "arc (358 全扰动), 种子42",
    "arc_s777": "arc (358 全扰动), 种子777",
    "mixed_s42": "mixed (190干净+190arc), 种子42",
}


def wilson(s, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m), min(1.0, c + m)


def load(tag):
    p = f"{RES}/{tag}.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def paired(a, b):
    """b - a on the trials both completed. Returns (n, diff, se, p_mcnemar, n01, n10)."""
    ra = {r["trial"]: r for r in a["records"]}
    rb = {r["trial"]: r for r in b["records"]}
    common = sorted(set(ra) & set(rb))
    n = len(common)
    n01 = sum(1 for t in common if ra[t]["success"] and not rb[t]["success"])
    n10 = sum(1 for t in common if not ra[t]["success"] and rb[t]["success"])
    diff = (n10 - n01) / n
    se = math.sqrt(n01 + n10 - (n10 - n01) ** 2 / n) / n if (n01 + n10) else 0.0
    disc = n01 + n10
    if disc:
        k = min(n01, n10)
        pv = min(1.0, 2 * sum(math.comb(disc, i) for i in range(k + 1)) / (2 ** disc))
    else:
        pv = 1.0
    return n, diff, se, pv, n01, n10


data = {k: load(k) for k in RUNS}
missing = [k for k, v in data.items() if v is None]

print("=" * 80)
print("各组成功率")
print("=" * 80)
for k, label in RUNS.items():
    d = data[k]
    if d is None:
        print(f"  {label:<34} (尚未完成)")
        continue
    lo, hi = wilson(d["n_success"], d["n_trials"])
    print(f"  {label:<34} {d['n_success']:>4}/{d['n_trials']:<4} = {d['success_rate']:.4f}  "
          f"[{lo:.4f}, {hi:.4f}]")

print()
print("=" * 80)
print("噪声底线: 同一批数据, 只换训练种子")
print("=" * 80)
floors = []
for grp in ["baseline", "arc"]:
    a, b = data.get(f"{grp}_s42"), data.get(f"{grp}_s777")
    if a is None or b is None:
        print(f"  {grp}: 尚未齐全")
        continue
    n, diff, se, pv, n01, n10 = paired(a, b)
    floors.append(abs(diff))
    print(f"  {grp:9} 种子42 -> 种子777: {diff*100:+.2f} pp   "
          f"(配对 n={n}, SE {se*100:.2f} pp, McNemar p={pv:.4f})")

if floors:
    fl = max(floors)
    print()
    print(f"  >>> 观测到的训练随机波动幅度: 最大 {fl*100:.2f} pp")
    print(f"      任何小于这个量级的数据集间差异, 都无法与训练噪声区分开")

print()
print("=" * 80)
print("三个数据集对比 (训练种子统一为 42)")
print("=" * 80)
base = data.get("baseline_s42")
for other, label in [("arc_s42", "arc"), ("mixed_s42", "mixed")]:
    d = data.get(other)
    if base is None or d is None:
        print(f"  {label}: 尚未齐全")
        continue
    n, diff, se, pv, n01, n10 = paired(base, d)
    lo95, hi95 = diff - 1.96 * se, diff + 1.96 * se
    lo90, hi90 = diff - 1.645 * se, diff + 1.645 * se
    margin = max(abs(lo90), abs(hi90))
    verdict = ""
    if floors:
        verdict = "  <= 在噪声底线之内, 无法归因于数据" if abs(diff) <= max(floors) else \
                  "  > 超出噪声底线"
    print(f"  {label} vs baseline: {diff*100:+.2f} pp   p={pv:.4f}   "
          f"95%CI [{lo95*100:+.2f}, {hi95*100:+.2f}]{verdict}")
    print(f"      等价界限 ±{margin*100:.1f} pp   (仅当认为 {margin*100:.0f}pp 以内算'一样'时才支持等价)")

print()
print("=" * 80)
print("对照: v1 (恒定学习率, 已作废)")
print("=" * 80)
print("  baseline@19999 243/300=0.8100   arc@19999 209/300=0.6967   差 -11.33pp  p=0.0011")
print("  baseline@16000 144/246=0.5854 <- 同一次训练内相隔4000步就差 22.5pp, 故该结论作废")

if missing:
    print()
    print(f"注意: 以下结果尚未产出 -> {missing}")
