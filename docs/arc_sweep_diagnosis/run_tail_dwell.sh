#!/usr/bin/env bash
# The dwell at the end of each subtask -- where placement happens -- rather than at the start.
#
# num_fixed_steps only dwells at subtask starts, which is why it fixed grasping (16 -> 11 and
# 31 -> 23 failures) and left placement untouched (154 -> 155), while placement is 66% of all
# failures. This appends the dwell to the other end.
set -u
T=/home/pk/.claude/jobs/10fee75c/tmp
PY=/home/pk/miniconda3/envs/isaaclab/bin/python
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PERTURB_FIXED_ATTEMPTS=1
cd "$T" || exit 1

# Wait on the other queue's completion marker, not on "no fix_trial.py running" -- that condition
# was also true for the subtask-gating queue at the moment the first batch ended, so both launched
# at once and two Isaac Sim instances contended for the same 12 GB card.
while ! grep -q ALL_GATE_DONE "$T/gate.log" 2>/dev/null; do sleep 20; done
while pgrep -f "fix_trial.py" > /dev/null; do sleep 20; done
sleep 10

run () {
  local tag="$1" arc="$2" tail="$3"
  echo "########## $tag  arc=$arc tail_dwell=$tail ##########"
  PERTURB_ARC_STD="$arc" TAIL_DWELL="$tail" \
    timeout 1800 "$PY" fix_trial.py --attempts 300 --num_envs 10 --headless --device cuda:0 \
      --output_file "$T/out/fix_$tag.hdf5" > "fix_$tag.log" 2>&1
  echo "  rc=$?"
  grep -aE "^\[fix\]|^\[cfg\]" "fix_$tag.log"
  grep -aoE "[0-9]+/[0-9]+ \([0-9.]+%\) successful" "fix_$tag.log" | tail -1
}

run tail_high 0.030 20
run tail_low  0.005 20
echo ALL_TAIL_DONE
