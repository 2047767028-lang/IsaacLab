#!/usr/bin/env bash
# Direction 2 at the contact frame: five runs, 300 fixed attempts each, one harness (num_envs 10,
# per-episode reseeding, PERTURB_STD left at the task default as in EXPERIMENT_LEDGER.md group D).
#
#   ref_none   arc 0.5 cm, no fix          group reference; source of the contact-pose table
#   arc_none   arc 3.0 cm, no fix          the penalty being attacked
#   arc_snap   arc 3.0 cm, snap_ref        the proposal: hold at the reference run's contact pose
#   arc_hold   arc 3.0 cm, hold_target     control: same hold, at the nominal target, no reference
#   ref_hold   arc 0.5 cm, hold_target     does the hold lift the low-amplitude reference too?
#
# Everything is parameterised so the same file runs on the laptop or the lab server:
#   PY       python of the Isaac Lab env      OUT     output directory (created)
#   INPUT    annotated source-demo hdf5       SCRIPTS directory holding contact_hold_trial.py etc.
#   DEVICE   cuda:N seen by the process       CUDA_VISIBLE_DEVICES may be set by the caller
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

run ref_none PERTURB_ARC_STD=0.005 CONTACT_FIX=none
"$PY" "$SCRIPTS/build_contact_table.py" "$OUT/ch_ref_none" "$OUT/ref_table.npz" 2>&1 | tail -2
run arc_none PERTURB_ARC_STD=0.030 CONTACT_FIX=none
run arc_snap PERTURB_ARC_STD=0.030 CONTACT_FIX=snap_ref REF_TABLE="$OUT/ref_table.npz"
run arc_hold PERTURB_ARC_STD=0.030 CONTACT_FIX=hold_target
run ref_hold PERTURB_ARC_STD=0.005 CONTACT_FIX=hold_target
say "ALL_CONTACT_HOLD_DONE"
