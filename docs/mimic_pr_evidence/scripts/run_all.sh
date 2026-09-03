#!/usr/bin/env bash
# Four closed-loop runs on the stock Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 task.
#
# A/B are the max_num_failures defect and its fix: A shows the shipped cap of 25 being ignored,
# B shows the same run stopping at the cap once the field is actually read.
#
# C/D are the cubes_stacked defect and its fix: C generates demos with the stock criterion and D
# with the at-rest requirement added. The measurement is the share of ACCEPTED demos whose stack is
# broken by the final frame -- those are episodes where the cube was dropped and the criterion fired
# on it mid-flight.
#
# Sequential on purpose: the runs share one GPU.
set -u
TMP=/home/pk/.claude/jobs/10fee75c/tmp
R="$TMP/run_validation.sh"
TASK=Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0
NE=10

echo "########## A: max_num_failures, stock loop (expect failures to blow past the cap) ##########"
MAX_NUM_FAILURES=25 FIX_MAX_FAILURES=0 \
  "$R" A --task "$TASK" --output_file "$TMP/out/A.hdf5" --generation_num_trials 30 --num_envs $NE \
  > "$TMP/A.log" 2>&1
grep -E "^RESULT |EXITCODE=" "$TMP/A.log"

echo "########## B: max_num_failures, patched loop (expect a stop at the cap) ##########"
MAX_NUM_FAILURES=25 FIX_MAX_FAILURES=1 \
  "$R" B --task "$TASK" --output_file "$TMP/out/B.hdf5" --generation_num_trials 30 --num_envs $NE \
  > "$TMP/B.log" 2>&1
grep -E "^RESULT |EXITCODE=|\[FIX\]" "$TMP/B.log"

echo "########## C: stock success criterion ##########"
FIX_MAX_FAILURES=0 \
  "$R" C --task "$TASK" --output_file "$TMP/out/C.hdf5" --generation_num_trials 100 --num_envs $NE \
  > "$TMP/C.log" 2>&1
grep -E "^RESULT |EXITCODE=" "$TMP/C.log"

echo "########## D: criterion with the at-rest requirement ##########"
FIX_MAX_FAILURES=0 FIX_AT_REST=0.01 \
  "$R" D --task "$TASK" --output_file "$TMP/out/D.hdf5" --generation_num_trials 100 --num_envs $NE \
  > "$TMP/D.log" 2>&1
grep -E "^RESULT |EXITCODE=|\[fix\]" "$TMP/D.log"

echo "########## ALL RUNS COMPLETE ##########"
