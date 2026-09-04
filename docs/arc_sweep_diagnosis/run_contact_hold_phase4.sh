#!/usr/bin/env bash
# Contact-frame hold, phase 4: how large can the arc go once the contact frame is gated?
# 300 fixed attempts each, num_envs 10, scene draw 1, freeze_frac 0.3, isotropic directions,
# gate_target with the cap raised to 60 steps so a large residual can still converge.
#
#   gt_zero   0 cm  (no arc at all) -- the clean "MimicGen + gate" reference
#   big5      5 cm
#   big8      8 cm
#   big10    10 cm
set -u
PY=${PY:-/home/pk/miniconda3/envs/isaaclab/bin/python}
OUT=${OUT:-/home/pk/.claude/jobs/23d24a02/tmp/ch_out}
INPUT=${INPUT:-/home/pk/IsaacLab/datasets/annotated_dataset.hdf5}
SCRIPTS=${SCRIPTS:-$(cd "$(dirname "$0")" && pwd)}
DEVICE=${DEVICE:-cuda:0}
ATTEMPTS=${ATTEMPTS:-300}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PERTURB_FIXED_ATTEMPTS=1 RESEED=1
export GATE_TOL=${GATE_TOL:-0.003} GATE_MAX=${GATE_MAX:-60}
mkdir -p "$OUT"
say () { echo "[$(date '+%F %T')] $*"; }

run () {
  local tag="$1"; shift
  say "########## $tag ##########"
  rm -f "$OUT/$tag.hits"
  env "$@" HITS_FILE="$OUT/$tag.hits" CMD_DIR="$OUT/cmd_$tag" timeout 3600 "$PY" "$SCRIPTS/contact_hold_trial.py" \
      --attempts "$ATTEMPTS" --num_envs 10 --headless --device "$DEVICE" \
      --input_file "$INPUT" --output_file "$OUT/ch_$tag.hdf5" > "$OUT/$tag.log" 2>&1
  say "  rc=$?  counters: $(cat "$OUT/$tag.hits" 2>/dev/null || echo n/a)"
  say "  $(grep -aoE "[0-9]+/[0-9]+ \([0-9.]+%\) successful" "$OUT/$tag.log" | tail -1)"
}

run gt_zero PERTURB_ARC_STD=0     CONTACT_FIX=gate_target
run big5    PERTURB_ARC_STD=0.050 CONTACT_FIX=gate_target
run big8    PERTURB_ARC_STD=0.080 CONTACT_FIX=gate_target
run big10   PERTURB_ARC_STD=0.100 CONTACT_FIX=gate_target
say "ALL_PHASE4_DONE"
