#!/usr/bin/env bash
# Contact-frame hold, phase 9: peak-position sweep at 10 cm, freeze 0.3, gated hold -- does an
# earlier return alone (no extra freeze) recover the yield, i.e. is "return time" separable from
# "less perturbation"? pk10 (peak 0.25) exists: 31.0%. The envelope family u^a (1-u)^b with a+b=6
# needs a > 1, so the peak cannot go below 1/6; 0.17 is the floor.
#
#   pk10_p17   10 cm, peak 0.17, freeze 0.3
#   pk10_p35   10 cm, peak 0.35, freeze 0.3
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

run pk10_p17 PERTURB_ARC_STD=0.100 CONTACT_FIX=gate_target PERTURB_ARC_PEAK_FRAC_MIN=0.17 PERTURB_ARC_PEAK_FRAC_MAX=0.17 PERTURB_ARC_FREEZE_FRAC=0.3
run pk10_p35 PERTURB_ARC_STD=0.100 CONTACT_FIX=gate_target PERTURB_ARC_PEAK_FRAC_MIN=0.35 PERTURB_ARC_PEAK_FRAC_MAX=0.35 PERTURB_ARC_FREEZE_FRAC=0.3
say "ALL_PHASE9_DONE"
