# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.utils import configclass

from .franka_stack_ik_rel_mimic_env_cfg import FrankaCubeStackIKRelMimicEnvCfg


@configclass
class FrankaCubeStackIKRelPerturbedMimicEnvCfg(FrankaCubeStackIKRelMimicEnvCfg):
    """Franka Cube Stack IK Rel Mimic env cfg with a sweepable reset-time joint noise magnitude.

    Reuses the existing ``randomize_franka_joint_state`` reset event (already present in the
    base task via ``stack_joint_pos_env_cfg.EventCfg``) but makes its ``std`` controllable via the
    ``PERTURB_STD`` environment variable, so the same registered env can be swept over multiple
    perturbation magnitudes across separate generation runs (one Isaac Sim process per magnitude).
    Defaults to 0.02, matching the fixed value used by the unperturbed Mimic env.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_franka_joint_state.params["std"] = float(os.environ.get("PERTURB_STD", "0.02"))
