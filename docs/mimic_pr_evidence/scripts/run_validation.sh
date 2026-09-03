#!/usr/bin/env bash
# Runner for the mimic-defect validation. Captures the exit code, which a silent death otherwise
# hides: the first attempt vanished during gym.make with no traceback in the log.
set -u

TMP=/home/pk/.claude/jobs/10fee75c/tmp
PY=/home/pk/miniconda3/envs/isaaclab/bin/python
LAB=/home/pk/IsaacLab

LABEL="$1"; shift

cd "$TMP" || exit 1
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

echo "=== $LABEL started $(date -Is) ==="
echo "=== env: FIX_MAX_FAILURES=${FIX_MAX_FAILURES:-unset} FIX_AT_REST=${FIX_AT_REST:-unset} MAX_NUM_FAILURES=${MAX_NUM_FAILURES:-unset} ==="
"$PY" "$TMP/validate_mimic_bugs.py" \
  --input_file "$LAB/datasets/annotated_dataset.hdf5" \
  --headless --device cuda:0 \
  "$@"
rc=$?
echo "=== $LABEL EXITCODE=$rc finished $(date -Is) ==="
