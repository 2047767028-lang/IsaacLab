"""Measure OBJ bounding boxes and vertex counts without external deps."""
import sys
import glob
import os


def measure(path):
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    nv = 0
    nf = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                nv += 1
                p = line.split()
                for i in range(3):
                    x = float(p[i + 1])
                    lo[i] = min(lo[i], x)
                    hi[i] = max(hi[i], x)
            elif line.startswith("f "):
                nf += 1
    size = [hi[i] - lo[i] for i in range(3)]
    return nv, nf, lo, hi, size


for pat in sys.argv[1:]:
    for path in sorted(glob.glob(pat)):
        nv, nf, lo, hi, size = measure(path)
        name = os.path.basename(path)
        print(f"{name:<28} v={nv:>7} f={nf:>7}  size(x,y,z) = "
              f"{size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f}   "
              f"min=({lo[0]:.3f},{lo[1]:.3f},{lo[2]:.3f})")
