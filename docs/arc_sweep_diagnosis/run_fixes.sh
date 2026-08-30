#!/usr/bin/env bash
# Six runs, fixed 300 attempts each, on the same task and seed the v2 sweep used.
#
# Reference points first (should land near the sweep's 35.1% and 19.1%), then the two remedies at
# the amplitude where the penalty is largest, then the remedies at the low amplitude to see whether
# they move the reference too.
set -u
T=/home/pk/.claude/jobs/10fee75c/tmp
PY=/home/pk/miniconda3/envs/isaaclab/bin/python
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PERTURB_FIXED_ATTEMPTS=1
cd "$T" || exit 1
mkdir -p out

run () {
  local tag="$1" arc="$2" dwell="$3" scale="$4"
  echo "########## $tag  arc=$arc dwell=$dwell scale=$scale ##########"
  PERTURB_ARC_STD="$arc" DWELL="$dwell" SCALE="$scale" \
    timeout 1800 "$PY" fix_trial.py --attempts 300 --num_envs 10 --headless --device cuda:0 \
      --output_file "$T/out/fix_$tag.hdf5" > "fix_$tag.log" 2>&1
  echo "  rc=$?"
  grep -aE "^\[fix\]|^\[cfg\]|^RESULT " "fix_$tag.log"
}

run ref_low   0.005 0 0
run ref_high  0.030 0 0
run dwell_high 0.030 20 0
run scale_high 0.030 0 1.0
run dwell_low  0.005 20 0
run scale_low  0.005 0 1.0
echo ALL_FIXES_DONE
