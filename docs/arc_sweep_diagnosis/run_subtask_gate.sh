#!/usr/bin/env bash
# The fourth remedy: perturb fewer subtasks instead of trying to undo the damage afterwards.
#
# The contact-phase residual accumulates across subtasks like a random walk -- 0.63, 0.96, 1.01,
# 1.28 cm at the four contact events for a 3.0 cm arc, against sqrt(k) predictions of 0.63, 0.89,
# 1.09, 1.26 -- so perturbing k of four should scale it by sqrt(k/4). One subtask halves it.
#
# Prediction, if the residual is what drives the failures: gating 3.0 cm down to a single subtask
# should recover roughly what halving the amplitude recovers, while keeping full amplitude where it
# is applied.
set -u
T=/home/pk/.claude/jobs/10fee75c/tmp
PY=/home/pk/miniconda3/envs/isaaclab/bin/python
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PERTURB_FIXED_ATTEMPTS=1
cd "$T" || exit 1

# do not compete with the first batch for the GPU
while pgrep -f "fix_trial.py" > /dev/null; do sleep 20; done
sleep 10

run () {
  local tag="$1" arc="$2" subs="$3"
  echo "########## $tag  arc=$arc subtasks=$subs ##########"
  PERTURB_ARC_STD="$arc" ARC_SUBTASKS="$subs" \
    timeout 1800 "$PY" fix_trial.py --attempts 300 --num_envs 10 --headless --device cuda:0 \
      --output_file "$T/out/fix_$tag.hdf5" > "fix_$tag.log" 2>&1
  echo "  rc=$?"
  grep -aE "^\[fix\]|^\[cfg\]" "fix_$tag.log"
  grep -aoE "[0-9]+/[0-9]+ \([0-9.]+%\) successful" "fix_$tag.log" | tail -1
}

run gate1_high 0.030 0
run gate2_high 0.030 0,2
echo ALL_GATE_DONE
