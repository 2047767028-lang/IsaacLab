# Description

`DataGenConfig.max_num_failures` is documented as *"Maximum number of failures allowed before stopping generation"* and eighteen shipped Mimic environment configs set it to `25`. Nothing reads it.

```console
$ git grep -c max_num_failures
source/isaaclab/isaaclab/envs/mimic_env_cfg.py:1          # the definition
source/isaaclab_mimic/isaaclab_mimic/envs/*.py:18         # eighteen assignments
```

No third entry: there is no read anywhere in the repository, and the field has been inert since Isaac Lab Mimic was introduced in #179. `env_loop` counts `num_failures` and then never consults it; its only termination is

```python
check_val = num_success if generation_guarantee else num_attempts
if check_val >= generation_num_trials:
```

With `generation_guarantee = True` (which those same eighteen configs also set), a task whose success rate is low retries without any bound, and the one knob that claims to stop that does nothing. On our own data generation this cost 337 attempts for 10 demos before we discovered the cap was not real.

### Reproduction, on a stock task

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --input_file ./datasets/annotated_dataset.hdf5 \
  --output_file /tmp/out.hdf5 \
  --generation_num_trials 30 --num_envs 10 --headless
```

`FrankaCubeStackIKRelMimicEnvCfg` sets `max_num_failures = 25`. Observed:

| | successes | failures | attempts |
|---|---|---|---|
| before | 30 | **50** | 80 |
| after (`max_num_failures = 25`) | 12 | **25** | 37 |

Before the change the run passed the configured cap and kept going, ending at twice it. After, it stops on the attempt that reaches 25 failures and says so:

```
Reached 25 failures (max_num_failures=25) after 12/30 successes. Exiting.
```

### On the default

Wiring the field up exactly as written would abort the primary documented workflow. At the ~36% success rate measured above, `--generation_num_trials 1000` needs on the order of 1800 failures, so a cap of 25 would end the run after roughly forty attempts. Since the eighteen assignments were inert when they were written and no current behaviour depends on them, they are removed here and the default becomes `None`, meaning no limit. **Runs behave exactly as they do today unless a limit is explicitly requested.**

If maintainers would rather the shipped configs keep an active cap, that is a one-line change per config and I am happy to make it — it just needs a value that suits large runs.

## Type of change

- Bug fix (non-breaking change which fixes an issue)

## Checklist

- [x] I have read and understood the contribution guidelines
- [ ] I have run the `pre-commit` checks with `./isaaclab.sh --format`
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [x] I have added a changelog fragment under `source/<pkg>/changelog.d/` for every touched package
- [ ] I have added my name to the `CONTRIBUTORS.md` or my name already exists there
