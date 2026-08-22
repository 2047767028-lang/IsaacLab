# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.utils import configclass

from .franka_stack_ik_rel_visuomotor_mimic_env_cfg import FrankaCubeStackIKRelVisuomotorMimicEnvCfg


@configclass
class FrankaCubeStackIKRelVisuomotorPerturbedMimicEnvCfg(FrankaCubeStackIKRelVisuomotorMimicEnvCfg):
    """Camera-equipped Franka Cube Stack IK Rel Mimic env cfg, wired to the same PERTURB_* env
    vars as FrankaCubeStackIKRelPerturbedMimicEnvCfg (the non-visual variant used for the v1/v2
    diagnostic sweeps).

    This wiring does NOT come for free just by using the visuomotor task: unlike
    PERTURB_ARC_STD/PERTURB_ARC_FREEZE_FRAC (read directly in data_generator.py, so they apply to
    *any* task), PERTURB_STD/PERTURB_SEED/PERTURB_FIXED_ATTEMPTS were only ever wired into
    FrankaCubeStackIKRelPerturbedMimicEnvCfg.__post_init__ - a *different* class in a *different*
    inheritance chain than FrankaCubeStackIKRelVisuomotorMimicEnvCfg. Found this the hard way
    during a smoke test for the pi0.5 production data run: with the plain visuomotor task,
    PERTURB_FIXED_ATTEMPTS was silently ignored (generation_guarantee stayed True, so
    "--generation_num_trials 20" kept retrying past 20 total attempts trying to bank 20
    *successes*), and PERTURB_STD was also silently ignored (the visuomotor EventCfg subclasses
    stack_joint_pos_env_cfg.EventCfg, which hardcodes randomize_franka_joint_state's std at 0.02
    with no env-var override) - so a "baseline, PERTURB_STD=0" group would have silently still
    carried 0.02 std reset joint noise. Both would have been wrong for data meant to actually be
    trained on. This class exists so the visuomotor task gets the exact same overrides as the
    non-visual one, rather than only some of them.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_franka_joint_state.params["std"] = float(os.environ.get("PERTURB_STD", "0.02"))
        self.datagen_config.seed = int(os.environ.get("PERTURB_SEED", "1"))
        if os.environ.get("PERTURB_FIXED_ATTEMPTS"):
            self.datagen_config.generation_guarantee = False
