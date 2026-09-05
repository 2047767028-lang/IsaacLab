import json
import math

RES = "/home/pk/.claude/jobs/2cbde0b3/tmp/final"


def load(t):
    return json.load(open(f"{RES}/{t}.json"))


def wilson(s, n, z=1.96):
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m), min(1.0, c + m)


runs = {t: load(t) for t in ["baseline_s42", "baseline_s777", "arc_s42", "arc_s777"]}

print("=" * 74)
print("四次运行")
print("=" * 74)
for t, d in runs.items():
    lo, hi = wilson(d["n_success"], d["n_trials"])
    print(f"  {t:<16} {d['n_success']:>3}/{d['n_trials']} = {d['success_rate']:.4f}  [{lo:.4f},{hi:.4f}]")

print()
print("=" * 74)
print("按种子平均 (每组2个种子, 汇总600次rollout)")
print("=" * 74)
agg = {}
for g in ["baseline", "arc"]:
    s = runs[f"{g}_s42"]["n_success"] + runs[f"{g}_s777"]["n_success"]
    n = runs[f"{g}_s42"]["n_trials"] + runs[f"{g}_s777"]["n_trials"]
    agg[g] = (s, n, s / n)
    lo, hi = wilson(s, n)
    print(f"  {g:<9} {s}/{n} = {s/n:.4f}  [{lo:.4f},{hi:.4f}]")

pb, pa = agg["baseline"][2], agg["arc"][2]
nb, na = agg["baseline"][1], agg["arc"][1]
diff = pa - pb
se = math.sqrt(pb * (1 - pb) / nb + pa * (1 - pa) / na)
print(f"\n  种子平均后的数据集差异 = {diff*100:+.2f} pp")
print(f"  SE {se*100:.2f} pp   95%CI [{(diff-1.96*se)*100:+.2f}, {(diff+1.96*se)*100:+.2f}] pp")
print("  对比: 只用种子42时是 -5.33 pp")

print()
print("=" * 74)
print("种子间差异本身显著吗")
print("=" * 74)
for g, d1, d2, pv in [("baseline", -0.0367, 0.0363, 0.3594), ("arc", 0.0333, 0.0388, 0.4404)]:
    print(f"  {g:<9} {d1*100:+.2f} pp  SE {d2*100:.2f}  p={pv:.4f}  "
          f"95%CI [{(d1-1.96*d2)*100:+.2f},{(d1+1.96*d2)*100:+.2f}]  -> 与0无异")
print("\n  => 两个种子差异都不显著, 所以严格讲只能说训练噪声 <= 约 4pp,")
print("     不能说'已测得训练噪声就是 3.5pp'。")
