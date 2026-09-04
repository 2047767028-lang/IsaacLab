#!/usr/bin/env bash
# Contact-frame hold, phase 2: robustness, approach-phase disturbance, and the production point.
# 300 fixed attempts each, num_envs 10, same harness as group E (EXPERIMENT_LEDGER.md).
#
#   s2_ref_hold   0.5 cm + hold 20, scene sequence base 2000000   } is 50.3% / 39.0% a property of
#   s2_arc_hold   3.0 cm + hold 20, scene sequence base 2000000   } one scene draw?
#   fz_ref_hold   0.5 cm + hold 20, freeze_frac 0.5               } does shortening the perturbed part of
#   fz_arc_hold   3.0 cm + hold 20, freeze_frac 0.5               } the approach shrink the residual gap?
#   op_arc_hold   1.2 cm + hold 20 (the production amplitude)     production yield with the hold
set -u
PY=${PY:-/home/pk/miniconda3/envs/isaaclab/bin/python}
OUT=${OUT:-/home/pk/.claude/jobs/23d24a02/tmp/ch_out}
INPUT=${INPUT:-/home/pk/IsaacLab/datasets/annotated_dataset.hdf5}
SCRIPTS=${SCRIPTS:-$(cd "$(dirname "$0")" && pwd)}
DEVICE=${DEVICE:-cuda:0}
ATTEMPTS=${ATTEMPTS:-300}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PERTURB_FIXED_ATTEMPTS=1 RESEED=1 HOLD=${HOLD:-20} RAMP=${RAMP:-10}
mkdir -p "$OUT"
say () { echo "[$(date '+%F %T')] $*"; }

run () {
  local tag="$1"; shift
  say "########## $tag ##########"
  rm -f "$OUT/$tag.hits"
  env "$@" HITS_FILE="$OUT/$tag.hits" timeout 2400 "$PY" "$SCRIPTS/contact_hold_trial.py" \
      --attempts "$ATTEMPTS" --num_envs 10 --headless --device "$DEVICE" \
      --input_file "$INPUT" --output_file "$OUT/ch_$tag.hdf5" > "$OUT/$tag.log" 2>&1
  say "  rc=$?  counters: $(cat "$OUT/$tag.hits" 2>/dev/null || echo n/a)"
  say "  $(grep -aoE "[0-9]+/[0-9]+ \([0-9.]+%\) successful" "$OUT/$tag.log" | tail -1)"
}

run s2_ref_hold PERTURB_ARC_STD=0.005 CONTACT_FIX=hold_target RESEED_BASE=2000000
run s2_arc_hold PERTURB_ARC_STD=0.030 CONTACT_FIX=hold_target RESEED_BASE=2000000
run fz_ref_hold PERTURB_ARC_STD=0.005 CONTACT_FIX=hold_target PERTURB_ARC_FREEZE_FRAC=0.5
run fz_arc_hold PERTURB_ARC_STD=0.030 CONTACT_FIX=hold_target PERTURB_ARC_FREEZE_FRAC=0.5
run op_arc_hold PERTURB_ARC_STD=0.012 CONTACT_FIX=hold_target
say "ALL_PHASE2_DONE"
