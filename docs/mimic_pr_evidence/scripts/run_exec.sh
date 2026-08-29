#!/usr/bin/env bash
# Runs exec_pr_code.py under the Omniverse app and leaves the full log next to it.
cd /home/pk/.claude/jobs/10fee75c/tmp || exit 1
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
timeout 300 /home/pk/miniconda3/envs/isaaclab/bin/python exec_pr_code.py /home/pk/IsaacLab/.claude/worktrees/mimic-bugfix > exec_pr_code.log 2>&1
echo "EXITCODE=$?" >> exec_pr_code.log
