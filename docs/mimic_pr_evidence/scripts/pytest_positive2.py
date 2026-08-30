"""Positive control, second attempt: run the PR's test bodies against the PR's generation.py.

The first attempt handed the file straight to pytest and hung: the test module launches Kit at
import time, and a second AppLauncher in a process that already has one does not return. In CI each
test file gets its own process, so the launch preamble is exactly right there and the hang is a
local artefact of doing the injection in-process.

So the preamble is stripped for this run and nothing else. The test bodies, the fixtures and the
assertions are the committed file byte for byte -- the diff is checked and printed below, and it is
only the AppLauncher stanza.
"""

import difflib
import os
import sys

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import importlib.util

import pytest

TMP = "/home/pk/.claude/jobs/10fee75c/tmp"
PR_GENERATION = f"{TMP}/generation_pr.py"
PR_TEST = f"{TMP}/test_generation_failure_cap.py"
STRIPPED = f"{TMP}/test_generation_failure_cap_nolaunch.py"

PREAMBLE = "from isaaclab.app import AppLauncher\n\n# launch omniverse app\nsimulation_app = AppLauncher(headless=True).app\n"

original = open(PR_TEST).read()
assert PREAMBLE in original, "launch preamble not found verbatim -- refusing to guess"
stripped = original.replace(PREAMBLE, "", 1)
open(STRIPPED, "w").write(stripped)

removed = [
    line
    for line in difflib.unified_diff(original.splitlines(), stripped.splitlines(), lineterm="")
    if line.startswith("-") and not line.startswith("---")
]
print(f"[strip] removed {len(removed)} lines:", [line for line in removed])

spec = importlib.util.spec_from_file_location("isaaclab_mimic.datagen.generation", PR_GENERATION)
module = importlib.util.module_from_spec(spec)
sys.modules["isaaclab_mimic.datagen.generation"] = module
spec.loader.exec_module(module)
print(f"[inject] isaaclab_mimic.datagen.generation <- {module.__file__}")

code = pytest.main([STRIPPED, "-q", "-p", "no:cacheprovider", "--no-header"])
print(f"PYTEST_RESULT={code}")
sys.stdout.flush()
os._exit(0)
