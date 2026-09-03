#!/usr/bin/env bash
# Re-run of the end-of-subtask dwell, with the noise bug fixed.
#
# The first attempt let the appended frames inherit the subtask's action_noise of 0.03 -- 3 cm per
# axis -- so the arm was shoved around faster than it could settle and never converged: 0.9% of
# frames within a centimetre of target, against 2.8% for the shipped start-of-subtask dwell, and no
# effect on success. The appended frames are now noise-free, as the shipped fixed segment is.
set -u
T=/home/pk/.claude/jobs/10fee75c/tmp
PY=/home/pk/miniconda3/envs/isaaclab/bin/python
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PERTURB_FIXED_ATTEMPTS=1
cd "$T" || exit 1
while pgrep -f "fix_trial.py" > /dev/null; do sleep 20; done
sleep 5

run () {
  local tag="$1" arc="$2" tail="$3"
  echo "########## $tag  arc=$arc tail_dwell=$tail (noise-free) ##########"
  PERTURB_ARC_STD="$arc" TAIL_DWELL="$tail" \
    timeout 1800 "$PY" fix_trial.py --attempts 300 --num_envs 10 --headless --device cuda:0 \
      --output_file "$T/out/fix_$tag.hdf5" > "fix_$tag.log" 2>&1
  echo "  rc=$?"
  grep -aoE "[0-9]+/[0-9]+ \([0-9.]+%\) successful" "fix_$tag.log" | tail -1
}

run tail2_high 0.030 20
run tail2_low  0.005 20
echo ALL_TAIL2_DONE
