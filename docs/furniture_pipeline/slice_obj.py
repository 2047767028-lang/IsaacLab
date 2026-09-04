"""Slice an OBJ along an axis and report per-slice cross-section extent.

Reveals mating features: a peg shows as a narrow slice band at one end,
a socket shows as a hole (detected here via radial vertex distribution).

Usage: python3 slice_obj.py <file.obj> <axis 0|1|2> [nslices]
"""
import sys

path = sys.argv[1]
axis = int(sys.argv[2]) if len(sys.argv) > 2 else 1
n = int(sys.argv[3]) if len(sys.argv) > 3 else 20

verts = []
with open(path, "r", errors="replace") as f:
    for line in f:
        if line.startswith("v "):
            p = line.split()
            verts.append((float(p[1]), float(p[2]), float(p[3])))

lo = min(v[axis] for v in verts)
hi = max(v[axis] for v in verts)
span = hi - lo
other = [i for i in range(3) if i != axis]

print(f"{path}")
print(f"  axis={axis}  range=[{lo:.4f}, {hi:.4f}]  span={span:.4f} m  verts={len(verts)}")
print(f"  {'slice':<6} {'pos(m)':>9} {'n':>6} {'w1(mm)':>9} {'w2(mm)':>9} {'r_min(mm)':>10} {'r_max(mm)':>10}")

# center in the two non-slicing axes, for radial stats
cx = (max(v[other[0]] for v in verts) + min(v[other[0]] for v in verts)) / 2
cy = (max(v[other[1]] for v in verts) + min(v[other[1]] for v in verts)) / 2

for k in range(n):
    a = lo + span * k / n
    b = lo + span * (k + 1) / n
    sl = [v for v in verts if a <= v[axis] < b or (k == n - 1 and v[axis] == hi)]
    if not sl:
        print(f"  {k:<6} {(a+b)/2:>9.4f} {0:>6}        --        --         --         --")
        continue
    w1 = max(v[other[0]] for v in sl) - min(v[other[0]] for v in sl)
    w2 = max(v[other[1]] for v in sl) - min(v[other[1]] for v in sl)
    rs = [(((v[other[0]] - cx) ** 2 + (v[other[1]] - cy) ** 2) ** 0.5) for v in sl]
    print(f"  {k:<6} {(a+b)/2:>9.4f} {len(sl):>6} {w1*1000:>9.2f} {w2*1000:>9.2f} "
          f"{min(rs)*1000:>10.2f} {max(rs)*1000:>10.2f}")
