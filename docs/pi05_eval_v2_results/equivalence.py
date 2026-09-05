"""How close are the two groups, stated as an interval rather than as a p-value.

A non-significant McNemar test does not say "no difference" -- it says the data could not rule one
out. The interval below is what the data actually constrains, and the equivalence margin is the
smallest difference we are entitled to claim the two groups fall within.
"""
import json
import math

TMP = "/home/pk/.claude/jobs/2cbde0b3/tmp"
b = json.load(open(f"{TMP}/v2_baseline.json"))
a = json.load(open(f"{TMP}/v2_arc.json"))
rb = {r["trial"]: r for r in b["records"]}
ra = {r["trial"]: r for r in a["records"]}
common = sorted(set(rb) & set(ra))
n = len(common)

n01 = sum(1 for t in common if rb[t]["success"] and not ra[t]["success"])   # baseline only
n10 = sum(1 for t in common if not rb[t]["success"] and ra[t]["success"])   # arc only
n11 = sum(1 for t in common if rb[t]["success"] and ra[t]["success"])
n00 = n - n01 - n10 - n11

pb = (n11 + n01) / n
pa = (n11 + n10) / n
diff = pa - pb

# paired difference standard error (McNemar / Agresti)
se = math.sqrt(n01 + n10 - (n10 - n01) ** 2 / n) / n

print("=" * 74)
print(f"配对样本 n={n}   都成功 {n11}  都失败 {n00}  仅baseline {n01}  仅arc {n10}")
print("=" * 74)
print(f"  baseline {pb:.4f}    arc {pa:.4f}    差异 = {diff*100:+.2f} pp")
print(f"  配对差异标准误 SE = {se*100:.2f} pp")
print()
for conf, z, label in [(0.95, 1.96, "95%"), (0.90, 1.645, "90%")]:
    lo, hi = diff - z * se, diff + z * se
    print(f"  {label} 置信区间: [{lo*100:+.2f}, {hi*100:+.2f}] pp")

lo90, hi90 = diff - 1.645 * se, diff + 1.645 * se
margin = max(abs(lo90), abs(hi90))
print()
print("=" * 74)
print("等价性结论 (TOST, α=0.05 → 看 90% 区间)")
print("=" * 74)
print(f"  数据支持的等价界限: 两组差异不超过 ±{margin*100:.1f} pp")
print(f"  也就是说, 只有当你认为 {margin*100:.0f}pp 以内算'性能几乎一样'时,")
print(f"  本实验才支持'两组等价'这个说法。")
print()
print("  想把等价界限收窄到给定值, 需要多少样本 (假设点估计不变):")
for target in [0.10, 0.09, 0.08]:
    need = target - abs(diff)
    if need <= 0:
        print(f"    ±{target*100:.0f} pp : 不可能 —— 点估计 {abs(diff)*100:.1f}pp 本身已超出该界限")
        continue
    se_need = need / 1.645
    factor = (se / se_need) ** 2
    print(f"    ±{target*100:.0f} pp : 需 n ≈ {int(n*factor)}  (当前的 {factor:.1f} 倍)")

print()
print("=" * 74)
print("换个说法: 数据与哪些真实差异相容")
print("=" * 74)
lo95, hi95 = diff - 1.96 * se, diff + 1.96 * se
print(f"  与数据相容的真实差异范围: arc 比 baseline 低 {abs(lo95)*100:.1f}pp  到  高 {hi95*100:.1f}pp")
print(f"  v1 曾声称的 -11.33 pp {'仍在' if lo95 <= -0.1133 <= hi95 else '已落在'}这个区间"
      f"{'内(未被排除)' if lo95 <= -0.1133 <= hi95 else '外(已被排除)'}")
print(f"  '完全无差异(0pp)' {'在' if lo95 <= 0 <= hi95 else '不在'}这个区间内")
