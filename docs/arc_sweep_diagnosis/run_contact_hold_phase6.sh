#!/usr/bin/env bash
# Contact-frame hold, phase 6: slow the bump instead of shrinking it. The free zone of every
# subtask is time-stretched 2x (each command frame repeated twice), gate_target with cap 60,
# freeze 0.3, isotropic directions, scene draw 1, commanded-path dump on.
#
#   st0    0 cm, stretch 2   control: does slowing the free zone alone change placement?
#   st5    5 cm, stretch 2   group G: 26.0% at stretch 1, achieved arc 1.88 / commanded 4.35 cm
#   st10  10 cm, stretch 2   group G: 6.0% at stretch 1, achieved arc 3.49 / commanded 8.34 cm
set -u
PY=${PY:-/home/pk/miniconda3/envs/isaaclab/bin/python}
OUT=${OUT:-/home/pk/.claude/jobs/23d24a02/tmp/ch_out}
INPUT=${INPUT:-/home/pk/IsaacLab/datasets/annotated_dataset.hdf5}
SCRIPTS=${SCRIPTS:-$(cd "$(dirname "$0")" && pwd)}
DEVICE=${DEVICE:-cuda:0}
ATTEMPTS=${ATTEMPTS:-300}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PERTURB_FIXED_ATTEMPTS=1 RESEED=1
export GATE_TOL=${GATE_TOL:-0.003} GATE_MAX=${GATE_MAX:-60} PERTURB_ARC_STRETCH=${PERTURB_ARC_STRETCH:-2}
mkdir -p "$OUT"
say () { echo "[$(date '+%F %T')] $*"; }

run () {
  local tag="$1"; shift
  say "########## $tag ##########"
  rm -f "$OUT/$tag.hits"
  env "$@" HITS_FILE="$OUT/$tag.hits" CMD_DIR="$OUT/cmd_$tag" timeout 4200 "$PY" "$SCRIPTS/contact_hold_trial.py" \
      --attempts "$ATTEMPTS" --num_envs 10 --headless --device "$DEVICE" \
      --input_file "$INPUT" --output_file "$OUT/ch_$tag.hdf5" > "$OUT/$tag.log" 2>&1
  say "  rc=$?  counters: $(cat "$OUT/$tag.hits" 2>/dev/null || echo n/a)"
  say "  $(grep -aoE "[0-9]+/[0-9]+ \([0-9.]+%\) successful" "$OUT/$tag.log" | tail -1)"
}

run st0  PERTURB_ARC_STD=0     CONTACT_FIX=gate_target
run st5  PERTURB_ARC_STD=0.050 CONTACT_FIX=gate_target
run st10 PERTURB_ARC_STD=0.100 CONTACT_FIX=gate_target
say "ALL_PHASE6_DONE"
