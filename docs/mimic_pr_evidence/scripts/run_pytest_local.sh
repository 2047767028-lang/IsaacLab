#!/usr/bin/env bash
# Try to run a PR-branch pytest file locally: develop's isaaclab_tasks / isaaclab_mimic packages
# from the worktree take precedence on PYTHONPATH, everything else comes from the 2.3.2 install.
# Whether this works at all is what the run establishes.
set -u
W=/home/pk/IsaacLab/.claude/worktrees/mimic-bugfix
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$W/source/isaaclab_tasks:$W/source/isaaclab_mimic"
cd /home/pk/.claude/jobs/10fee75c/tmp || exit 1
timeout 400 /home/pk/miniconda3/envs/isaaclab/bin/python -m pytest "$1" -q -p no:cacheprovider > "pytest_$(basename "$1" .py).log" 2>&1
echo "PYTEST_EXITCODE=$?" >> "pytest_$(basename "$1" .py).log"
