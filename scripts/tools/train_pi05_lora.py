#!/usr/bin/env python3
"""pi05 LoRA training entrypoint for the converted LeRobot datasets.

Runs on the lab server against openpi; paths below are that machine's. Kept in this repo so the
training hyperparameters used for the perturbation study are version-controlled alongside the data
generation code they are compared against.

LEARNING RATE -- the v1 runs of this study were invalidated by getting this wrong, so the reasoning
is recorded here rather than left to the config values. v1 used:

    CosineDecaySchedule(warmup_steps=0, peak_lr=5e-5, decay_steps=10, decay_lr=5e-5)

with `peak_lr == decay_lr`, so the cosine ran from 5e-5 to 5e-5: a constant learning rate, twice
openpi's default peak, with no warmup and no annealing at all (openpi even logs "this results in a
constant schedule"). The models never entered a convergence phase -- `param_norm` climbed by an
identical +1.20 in every 5000-step window right through step 20000, and closed-loop success rate
swung 22.5pp (58.5% -> 81.0%) between the step-16000 and step-19999 checkpoints of the SAME run on
the SAME data. That wobble was twice the 11.3pp difference being attributed to the training data,
which made every cross-dataset comparison meaningless.

The defaults here follow openpi's own recipe with the decay stretched over the whole run, so the
learning rate actually reaches its floor before training stops.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path


REPO_ROOT = Path("/home/surf2/dev/mimic_enhance")
OPENPI_ROOT = Path("/home/surf2/dev/geniesim/genie_sim/openpi")
DATA_ROOT = REPO_ROOT / "lerobot_datasets"


def main() -> None:
    cuda_visible_devices = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    fsdp_devices = int(os.environ.get("PI05_SMOKE_FSDP_DEVICES", "1"))
    batch_size = int(os.environ.get("PI05_SMOKE_BATCH_SIZE", "32"))
    num_train_steps = int(os.environ.get("PI05_SMOKE_NUM_STEPS", "20000"))
    save_interval = int(os.environ.get("PI05_SMOKE_SAVE_INTERVAL", "4000"))
    repo_id = os.environ.get("PI05_SMOKE_REPO_ID", "pi05_lerobot_baseline")
    config_name = os.environ.get("PI05_SMOKE_CONFIG_NAME", "pi05_mimicgen_baseline_smoke")
    exp_name = os.environ.get("PI05_SMOKE_EXP_NAME", f"lora_{fsdp_devices}gpu_b{batch_size}_{num_train_steps}")

    # Re-running a condition under a different seed is what turns a single success-rate number into
    # an interpretable one: without knowing how much two identically-configured runs differ, a gap
    # between two *datasets* cannot be told apart from ordinary run-to-run variation.
    train_seed = int(os.environ.get("PI05_TRAIN_SEED", "42"))

    warmup_steps = int(os.environ.get("PI05_WARMUP_STEPS", "1000"))
    peak_lr = float(os.environ.get("PI05_PEAK_LR", "2.5e-5"))
    decay_lr = float(os.environ.get("PI05_DECAY_LR", "2.5e-6"))
    # Anneal across the whole run by default. Leaving this short is exactly what broke v1: the
    # schedule finishes decaying long before training stops, so the run spends its entire length
    # at a flat high learning rate and never settles.
    decay_steps = int(os.environ.get("PI05_DECAY_STEPS", str(num_train_steps)))

    # Resume an interrupted run from the latest checkpoint in its directory instead of wiping it.
    # openpi refuses resume+overwrite together, so both are tied to this one switch. What a resume
    # restores: params, optimizer state and the step counter, so the cosine schedule continues from
    # where it stopped. What it does NOT restore: the data iterator (openpi's `restore_state`
    # discards the data loader), so the resumed run re-draws batches from the start of the shuffle
    # stream. That is run-to-run variation of the same kind as changing the training seed, not a
    # change of hyperparameters -- record it alongside the result, don't treat it as a new condition.
    resume = os.environ.get("PI05_RESUME", "0") == "1"

    os.environ.setdefault("HF_LEROBOT_HOME", str(DATA_ROOT))
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.chdir(OPENPI_ROOT)

    import openpi.models.pi0 as pi0
    import openpi.training.config as config_lib
    import openpi.training.optimizer as optimizer
    import openpi.training.weight_loaders as weight_loaders
    from scripts import train

    model_cfg = pi0.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_horizon=10,
        discrete_state_input=False,
    )

    base = config_lib.get_config("pi05_libero")
    cfg = dataclasses.replace(
        base,
        name=config_name,
        exp_name=exp_name,
        seed=train_seed,
        model=model_cfg,
        data=config_lib.LeRobotLiberoDataConfig(
            repo_id=repo_id,
            base_config=config_lib.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/pi05_base/params"),
        freeze_filter=model_cfg.get_freeze_filter(),
        ema_decay=None,
        batch_size=batch_size,
        num_workers=0,
        num_train_steps=num_train_steps,
        log_interval=1,
        save_interval=save_interval,
        keep_period=None,
        fsdp_devices=fsdp_devices,
        wandb_enabled=False,
        overwrite=not resume,
        resume=resume,
        checkpoint_base_dir=str(REPO_ROOT / "openpi_smoke_checkpoints"),
        assets_base_dir=str(REPO_ROOT / "openpi_smoke_assets"),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,
            decay_steps=decay_steps,
            decay_lr=decay_lr,
        ),
    )
    print(
        f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} fsdp_devices={fsdp_devices} "
        f"repo_id={repo_id} config_name={config_name} exp_name={exp_name}\n"
        f"batch_size={batch_size} num_train_steps={num_train_steps} save_interval={save_interval} "
        f"train_seed={train_seed} resume={resume}\n"
        f"lr: warmup={warmup_steps} peak={peak_lr:g} decay_steps={decay_steps} floor={decay_lr:g}",
        flush=True,
    )
    train.main(cfg)


if __name__ == "__main__":
    main()
