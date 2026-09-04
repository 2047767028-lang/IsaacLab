#!/usr/bin/env bash
# Contact-frame hold, phase 7: finish the bump's return before the frozen segment (direction
# decision of 2026-09-04, CLAUDE.md 2.17). The arc stays on the whole trajectory and the peak
# amplitude is not cut; what moves is WHEN the return ends (envelope peak at 25% of the free zone
# instead of 50%, via the v3 PERTURB_ARC_PEAK_FRAC knob, never used before) and how long the
# frozen segment is. gate_target, cap 60, isotropic, scene draw 1, commanded-path dump on.
#
#   pk5    5 cm, peak 0.25, freeze 0.3   control is big5 (peak 0.5, freeze 0.3): 26.0%, seat at grasp 0.85 cm
#   pk5f   5 cm, peak 0.25, freeze 0.5   both knobs
#   pk10f 10 cm, peak 0.25, freeze 0.5   control is big10: 6.0%, seat at grasp 1.85 cm
# Acceptance: seat at grasp back near the 0 cm run's 0.61 cm; then success; achieved arc reported.
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
  env "$@" HITS_FILE="$OUT/$tag.hits" CMD_DIR="$OUT/cmd_$tag" timeout 4200 "$PY" "$SCRIPTS/contact_hold_trial.py" \
      --attempts "$ATTEMPTS" --num_envs 10 --headless --device "$DEVICE" \
      --input_file "$INPUT" --output_file "$OUT/ch_$tag.hdf5" > "$OUT/$tag.log" 2>&1
  say "  rc=$?  counters: $(cat "$OUT/$tag.hits" 2>/dev/null || echo n/a)"
  say "  $(grep -aoE "[0-9]+/[0-9]+ \([0-9.]+%\) successful" "$OUT/$tag.log" | tail -1)"
}

run pk5   PERTURB_ARC_STD=0.050 CONTACT_FIX=gate_target PERTURB_ARC_PEAK_FRAC_MIN=0.25 PERTURB_ARC_PEAK_FRAC_MAX=0.25 PERTURB_ARC_FREEZE_FRAC=0.3
run pk5f  PERTURB_ARC_STD=0.050 CONTACT_FIX=gate_target PERTURB_ARC_PEAK_FRAC_MIN=0.25 PERTURB_ARC_PEAK_FRAC_MAX=0.25 PERTURB_ARC_FREEZE_FRAC=0.5
run pk10f PERTURB_ARC_STD=0.100 CONTACT_FIX=gate_target PERTURB_ARC_PEAK_FRAC_MIN=0.25 PERTURB_ARC_PEAK_FRAC_MAX=0.25 PERTURB_ARC_FREEZE_FRAC=0.5
say "ALL_PHASE7_DONE"
