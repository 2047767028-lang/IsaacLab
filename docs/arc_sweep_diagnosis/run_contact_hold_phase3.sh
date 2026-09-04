#!/usr/bin/env bash
# Contact-frame hold, phase 3: the convergence-gated form. Waits for phase 2's completion marker
# (never on a shared idle condition -- see EXPERIMENT_LEDGER.md "Standing procedure").
#
#   gt_ref   0.5 cm, gate_target: repeat the pre-transition frame until |target-achieved| < 0.3 cm
#            or 40 steps, then act
#   gt_arc   3.0 cm, gate_target
set -u
PY=${PY:-/home/pk/miniconda3/envs/isaaclab/bin/python}
OUT=${OUT:-/home/pk/.claude/jobs/23d24a02/tmp/ch_out}
INPUT=${INPUT:-/home/pk/IsaacLab/datasets/annotated_dataset.hdf5}
SCRIPTS=${SCRIPTS:-$(cd "$(dirname "$0")" && pwd)}
DEVICE=${DEVICE:-cuda:0}
ATTEMPTS=${ATTEMPTS:-300}
WAIT_FOR=${WAIT_FOR:-$OUT/run_phase2.log}
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PERTURB_FIXED_ATTEMPTS=1 RESEED=1
export GATE_TOL=${GATE_TOL:-0.003} GATE_MAX=${GATE_MAX:-40}
mkdir -p "$OUT"
say () { echo "[$(date '+%F %T')] $*"; }

if [ -n "$WAIT_FOR" ]; then
  say "waiting for ALL_PHASE2_DONE in $WAIT_FOR"
  until grep -q "ALL_PHASE2_DONE" "$WAIT_FOR" 2>/dev/null; do sleep 30; done
  sleep 10
fi

run () {
  local tag="$1"; shift
  say "########## $tag ##########"
  rm -f "$OUT/$tag.hits"
  env "$@" HITS_FILE="$OUT/$tag.hits" timeout 3000 "$PY" "$SCRIPTS/contact_hold_trial.py" \
      --attempts "$ATTEMPTS" --num_envs 10 --headless --device "$DEVICE" \
      --input_file "$INPUT" --output_file "$OUT/ch_$tag.hdf5" > "$OUT/$tag.log" 2>&1
  say "  rc=$?  counters: $(cat "$OUT/$tag.hits" 2>/dev/null || echo n/a)"
  say "  $(grep -aoE "[0-9]+/[0-9]+ \([0-9.]+%\) successful" "$OUT/$tag.log" | tail -1)"
}

run gt_ref PERTURB_ARC_STD=0.005 CONTACT_FIX=gate_target
run gt_arc PERTURB_ARC_STD=0.030 CONTACT_FIX=gate_target
say "ALL_PHASE3_DONE"
