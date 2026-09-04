#!/usr/bin/env bash
# Convert FurnitureBench square_table parts (MIT licensed) from OBJ to USD.
#
# SDF collision is not a preference here: the legs carry screw threads and the
# top carries matching sockets, and convexHull/convexDecomposition erase both,
# which would make the assembly physically impossible to simulate.
set -euo pipefail

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y

ISAACLAB_REPO=${ISAACLAB_REPO:-/home/pk/IsaacLab}
PYTHON=${PYTHON:-/home/pk/miniconda3/envs/isaaclab/bin/python}
SRC=${SRC:-/home/pk/.claude/jobs/aae45b91/tmp/fb/x/furniture-bench-main/furniture_bench/assets/furniture/mesh/square_table}
OUT=${OUT:-/home/pk/furniture_assembly/assets/square_table}

mkdir -p "$OUT"

# Masses are estimates for the 3D-printed toy-scale parts (PLA, mostly hollow).
# top is 162.5 x 31.2 x 162.5 mm; each leg is 30 x 87.5 x 30 mm.
convert() {
  local name=$1 mass=$2
  echo "=== converting ${name} (mass=${mass} kg) ==="
  "$PYTHON" "$ISAACLAB_REPO/scripts/tools/convert_mesh.py" \
    "$SRC/${name}.obj" "$OUT/${name}.usd" \
    --collision-approximation sdf --mass "$mass" --headless
}

convert square_table_top 0.15
for i in 1 2 3 4; do
  convert "square_table_leg${i}" 0.03
done

echo
echo "=== output ==="
ls -la "$OUT"
