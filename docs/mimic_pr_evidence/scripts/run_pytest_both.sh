#!/usr/bin/env bash
# Positive and negative control for the max_num_failures test file.
#
# positive: the PR branch's generation.py is injected under the package name, so the test exercises
#           the fix and must pass in full.
# negative: nothing injected, so the installed 2.3.2 loop runs and the cap-dependent cases must fail.
set -u
T=/home/pk/.claude/jobs/10fee75c/tmp
PY=/home/pk/miniconda3/envs/isaaclab/bin/python
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
cd "$T" || exit 1

echo "########## POSITIVE CONTROL: PR branch generation.py ##########"
PR_GENERATION_PY="$T/generation_pr.py" timeout 400 "$PY" -m pytest test_generation_failure_cap.py -q -p no:cacheprovider 2>&1 | tail -12

echo
echo "########## NEGATIVE CONTROL: installed 2.3.2 generation.py ##########"
timeout 400 "$PY" -m pytest test_generation_failure_cap.py -q -p no:cacheprovider 2>&1 | tail -12
