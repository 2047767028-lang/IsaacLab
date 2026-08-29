"""Can develop's isaaclab_tasks be imported on top of the installed 2.3.2 isaaclab?

Decides whether the pytest files can be run locally or only in upstream CI.
"""

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import os

try:
    from isaaclab_tasks.contrib.stack.mdp.terminations import cubes_stacked

    print("TASKS_IMPORT_OK", cubes_stacked.__module__)
except Exception as e:  # noqa: BLE001
    print("TASKS_IMPORT_FAIL", type(e).__name__, str(e)[:300])

try:
    from isaaclab_mimic.datagen import generation

    print("MIMIC_IMPORT_OK", generation.__file__)
except Exception as e:  # noqa: BLE001
    print("MIMIC_IMPORT_FAIL", type(e).__name__, str(e)[:300])

os._exit(0)
