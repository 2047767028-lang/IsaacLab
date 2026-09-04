"""Detect sockets/holes in a slab mesh by clustering vertices in a thin slice.

Usage: python3 find_holes.py <file.obj> <axis> <slice_lo> <slice_hi>
Prints 2D clusters of vertices in that slab, which reveal hole rims.
"""
import sys

path, axis = sys.argv[1], int(sys.argv[2])
slo, shi = float(sys.argv[3]), float(sys.argv[4])

verts = []
with open(path, "r", errors="replace") as f:
    for line in f:
        if line.startswith("v "):
            p = line.split()
            verts.append((float(p[1]), float(p[2]), float(p[3])))

other = [i for i in range(3) if i != axis]
sl = [(v[other[0]], v[other[1]]) for v in verts if slo <= v[axis] <= shi]
print(f"{path}: {len(sl)} verts in slab [{slo}, {shi}] on axis {axis}")
if not sl:
    sys.exit()

# grid-based clustering
cell = 0.004  # 4 mm
grid = {}
for x, y in sl:
    grid.setdefault((round(x / cell), round(y / cell)), []).append((x, y))

seen = set()
clusters = []
for k in grid:
    if k in seen:
        continue
    stack, comp = [k], []
    seen.add(k)
    while stack:
        c = stack.pop()
        comp.extend(grid[c])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = (c[0] + dx, c[1] + dy)
                if nb in grid and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    clusters.append(comp)

clusters.sort(key=len, reverse=True)
print(f"clusters: {len(clusters)}")
for i, c in enumerate(clusters[:12]):
    xs = [p[0] for p in c]
    ys = [p[1] for p in c]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    w = (max(xs) - min(xs)) * 1000
    h = (max(ys) - min(ys)) * 1000
    rs = [(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5) * 1000 for x, y in c]
    print(f"  #{i} n={len(c):>5} center=({cx*1000:>7.1f},{cy*1000:>7.1f})mm "
          f"bbox={w:>6.1f}x{h:>6.1f}mm  r=[{min(rs):.1f}, {max(rs):.1f}]mm")
