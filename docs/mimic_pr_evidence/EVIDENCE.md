# Evidence for the two Isaac Lab Mimic fixes

Everything below is reproducible from the scripts in this directory. Where a claim rests on a
replayed criterion, the replay was validated against known positives first — a reproduction that has
not been shown to pass demos the generator itself accepted is not evidence about the ones it
rejected.

## Defect 1 — `max_num_failures` is never read

**Static evidence**

```console
$ git grep -c max_num_failures origin/develop
source/isaaclab/isaaclab/envs/mimic_env_cfg.py:1        # definition
source/isaaclab_mimic/isaaclab_mimic/envs/*.py:18       # eighteen assignments, no reads
```

There is no third line. `env_loop`'s only termination is `check_val >= generation_num_trials` where
`check_val` is `num_success` or `num_attempts`; `num_failures` is incremented and never consulted.
Present since `5cf3d6185` (#179, the commit that introduced Isaac Lab Mimic), so it is not a
regression.

**Closed-loop evidence** — stock `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0`, `max_num_failures=25`,
`--generation_num_trials 30 --num_envs 10`:

| run | loop | successes | failures | attempts |
|---|---|---|---|---|
| A | stock | 30 | **50** | 80 |
| B | field honoured | 12 | **25** | 37 |

A passed the configured cap and ended at twice it. B stopped on the attempt that reached 25 and
printed `[FIX] Reached 25 failures (cap 25). Exiting.`

**Why the shipped `= 25` lines are removed rather than left active**: at the ~36% success rate
measured here, `--generation_num_trials 1000` needs roughly 1800 failures. A live cap of 25 would
end that run after about forty attempts, i.e. it would break the primary documented workflow. The
assignments were inert when written, so removing them preserves today's behaviour exactly and makes
the field opt-in.

## Defect 2 — `cubes_stacked` fires on a cube in flight

**What it is**: the check tests an instantaneous configuration with no requirement that the cubes be
supported or stationary. A cube released above its target satisfies it for a frame or two while
falling, and the generator's `generated_success = generated_success or exec_success` promotes that
single frame to a successful episode, which is then written into the dataset.

**Evidence on delivered datasets** (offline replay, validated at 100% reproduction of the
generator's verdict on all accepted demos):

| dataset | accepted | stack broken at final frame |
|---|---|---|
| baseline | 380 | 28 (7.4%) |
| arc_1p2cm | 358 | 19 (5.3%) |

Those demos end with the top cube a median 6.3 cm away in xy and at the same height as the cube it
should sit on. Speed at the qualifying frame: median 0.028–0.032 m/s versus 0.0000 m/s for the rest.

**One timeline** (`demo_336`, the case most favourable to a "tolerance too tight" reading):

```
frame  stacked  cube3-cube2 dz  cube3 z   jaw error
  191            7.91 cm        14.62 cm   18.465 mm   held above the target
  193            5.60 cm        12.33 cm    3.124 mm   jaws opening, cube falling
  195     *      5.01 cm        11.76 cm    0.424 mm   criterion fires, mid-fall
  196     *      4.30 cm        11.04 cm    0.135 mm
  199            0.33 cm         4.59 cm    0.012 mm   cube reaches the table
  205            0.00 cm         2.05 cm    0.007 mm   6.8 cm from cube_2
```

**Fresh closed-loop reproduction** — stock task, 100 accepted demos: **9/100 (9.0%)** end with the
stack broken, median final xy 5.99 cm, speed discriminator 0.0046 vs 0.0000 m/s.

**Threshold selection** — replayed from `states/rigid_object/*/root_velocity`, the same signal the
fix reads (replay validated at 100/100 reproduction):

| population | p10 | p50 | p90 | max |
|---|---|---|---|---|
| intact (n=91) | 0.0084 | **0.0132** | 0.0211 | **0.0341** |
| broken (n=9) | 0.0278 | **0.1042** | 0.5011 | 0.5629 |

Cubes resting on the stack are **not** at zero — the contact solver leaves about 0.013 m/s of
jitter. This is why the first threshold was wrong.

| max_lin_vel | keeps intact | keeps broken |
|---|---|---|
| 0.010 | 15/91 | 0/9 |
| 0.020 | 77/91 | 0/9 |
| 0.025 | 83/91 | 0/9 |
| **0.050** | **91/91** | **2/9** |
| 0.100 | 91/91 | 4/9 |

0.05 was chosen: it costs no genuine demonstrations and removes 7 of the 9 defects. It is
deliberately not a perfect separator — the distributions overlap at the bottom (intact max 0.0341,
the two surviving defects at 0.0278 and 0.0283), because a cube near the apex of a bounce is briefly
slow. Adding an angular-velocity term buys one more defect at the cost of four sound demos, so it
was left out.

**End-to-end confirmation** — stock task, 100 accepted demos each:

| run | criterion | accepted | ends broken | attempts | success rate |
|---|---|---|---|---|---|
| C | stock | 100 | **9 (9.0%)** | 279 | 35.8% |
| D | + at rest, 0.05 m/s | 100 | **2 (2.0%)** | 300 | 33.3% |

The residual 2% is the overlap the threshold sweep predicts (2 of 9 defects survive at 0.05), and
the 2.5 pp of success rate is the cost of no longer counting drops. An earlier run of D at
0.01 m/s — the value taken from the finite-difference proxy — came in at 6.8%, which is what sent
the threshold back to be measured against the signal the check actually reads.

## What was checked and rejected

The recorded claim that 11% of failed attempts are false negatives from a too-tight gripper
tolerance (0.104 mm on a 40 mm jaw) **does not hold**, and acting on it would have corrupted the
dataset. All 76 such episodes are drops: none has its stack intact at the final frame, and the top
cube has moved a median 10.8 cm by then. Relaxing the tolerance to 3 mm would have admitted 54 of
them as successful demonstrations. The tight tolerance masks defect 2 by accident; it does not cause
a defect of its own, and it is left untouched.

## Scripts

| script | what it establishes |
|---|---|
| `verify_criterion.py` | replays `cubes_stacked` offline; validates at 100% on accepted demos |
| `settling_check.py` | the rejected episodes reach fully open jaws — rules out a stuck gripper |
| `timing_diag.py` | geometry and release never coincide; geometry holds a median 4 frames |
| `final_state.py` | quantifies the end state: cube on the table, ~10.8 cm of drift |
| `one_episode.py` | the frame-by-frame timeline above |
| `at_rest_check.py` | the velocity discriminator, rejected vs accepted populations |
| `candidate_fixes.py`, `fix_sweep.py` | offline comparison of candidate fixes |
| `check_alignment.py` | `states` lags `obs` by one frame; needed before trusting a states replay |
| `threshold_v2.py` | threshold sweep from the signal the fix actually reads |
| `angular_check.py` | why angular velocity was not added |
| `validate_mimic_bugs.py`, `run_all.sh` | the closed-loop harness for runs A–D |
| `analyze_runs.py` | defect rate in a freshly generated dataset |

## Independent review pass (second model, 2026-08-29)

Everything above was produced in one session. This section is a second pass that started from the
committed PR branches and re-derived each claim, looking for the ways the first pass could have
fooled itself.

### What the first pass had not done

**The PR branches' own files had never been executed.** The closed-loop runs A–D validated the
*logic* of both fixes, but they did so by injecting equivalent code into the installed 2.3.2 tree.
The develop-based `terminations.py` and `generation.py` that a reviewer would actually read were
only ever `py_compile`d.

Closed by `scripts/exec_pr_code.py`: it materialises `git show <branch>:<path>` for both the PR
branch and untouched `origin/develop`, loads each by path under the Omniverse app, and drives them
with mock scene objects / a scripted fake environment. **23/23 checks pass** (`exec_pr_code.log`):

| what | stock develop | PR branch |
|---|---|---|
| resting three-cube stack | True | True |
| cube_3 falling through the stack at 0.10 m/s | **True (defect)** | False |
| same, `max_lin_vel=None` | — | True (opt-out) |
| cube_1 / cube_2 moving instead | — | False / False |
| 0.030 / 0.049 / 0.051 m/s | — | True / True / False |
| `cube_3_cfg=None`, third object moving | — | True (not consulted) |
| Franka-style remap `(cube_2, cube_3, None)`, top falling | — | False |
| batch of 4 with envs 1 and 3 moving | **[T,T,T,T]** | [T,F,T,F] |
| closed gripper on a resting stack | — | False |
| `env_loop`, cap=5, every attempt fails | **still running at 41 steps** | exits at 5 failures |
| `env_loop`, cap=None, every attempt fails | still running at 41 | still running at 41 (unchanged) |
| `env_loop`, 3 successes arrive before cap | — | exits on successes |
| `env_loop`, `generation_guarantee=False` | — | exits on 10 attempts (unchanged) |

### Discrepancy between the validation harness and the PR, checked and closed

The harness's at-rest wrapper checked `cube_2` and `cube_3`; the PR's default success term checks
`cube_1`, `cube_2` and `cube_3`. If the bottom cube jittered more than the others the PR would be
stricter than what run D validated. Measured on run C's recorded `root_velocity` at the qualifying
frame: `cube_1` intact max **0.0177 m/s** (lower than the other two), and the PR's three-cube
criterion at 0.05 keeps **91/91** intact and **2/9** broken — the same numbers as the harness.

### Things that would have changed the PR, checked

- **All six `cubes_stacked` call sites are `DoneTerm` success terminations.** The Mimic subtask
  boundary signals use a different function (`object_stacked`, observations side), so the at-rest
  requirement does not move annotation boundaries. `object_stacked` has the same one-sided geometry
  and could fire in flight too; it is consumed edge-triggered by annotation and is left out of scope.
- **`root_lin_vel_w` returns `ProxyArray` on develop and `ProxyArray.torch` exists** (`utils/warp/proxy_array.py:122`), matching the `root_pos_w.torch` already used in the same function.
- **The stack task's default physics backend on develop is PhysX** (`stack_env_cfg.py: default = isaacsim_physx`), the backend the 0.05 threshold was measured on. A Newton solver is offered as an alternative config; its resting jitter was not measured. Flagged in the PR body.
- **`int | None = None` on a configclass field** has precedent in the same package (`direct_rl_env_cfg.py: seed: int | None = None`).
- **The `contrib/stack/` path does not exist on `release/3.0.0-beta2`** (96 insertions = the whole file on develop), so `develop` is the only possible base; the two earlier XR PRs from this fork also target `develop` (`gh pr view 7380`).
- **The existing generation test** asserts only `"successes/attempts. Exiting"` in the output with `generation_num_trials=1`; neither change alters that path.
- **SO-101 stack task** uses `cubes_stacked` with the default `max_lin_vel`. It shares the cube assets with the Franka task; its jitter was not measured separately. Flagged.

### Tests added to the PR branches

- `source/isaaclab_tasks/test/contrib/stack/test_cubes_stacked_at_rest.py` (10 cases)
- `source/isaaclab_mimic/test/test_generation_failure_cap.py` (6 cases)

Both follow the repository convention (`AppLauncher(headless=True)` at the top, no simulator scene).
**They cannot be run natively on this machine**: the editable 2.3.2 install registers a
`sys.meta_path` finder that wins over `PYTHONPATH`, so `isaaclab_tasks.contrib` resolves to the
2.3.2 package (no `contrib/`) and `isaaclab_mimic.datagen.generation` to the 2.3.2 file. Their
function bodies are what `exec_pr_code.py` executed; the import lines themselves are exercised only
by upstream CI.

**Negative control** (`pytest_test_generation_failure_cap.log`): run against the unpatched 2.3.2
`env_loop`, the mimic test gives **2 failed, 4 passed** — exactly the two cap-dependent cases fail
(`'fuse' == 'exited'`; `(3, 4) == (2, 4)`, the unpatched loop overran the cap and collected a third
success), the four asserting unchanged behaviour pass. The test detects the defect it is written for.

### Not done

- The lab server (Isaac Lab 3.0.0) was considered for running the PR code end to end, but the
  `contrib/stack` layout does not exist there either, and repeated SSH attempts were refused mid-way
  (the host runs fail2ban) — dropped rather than hammered.
- `./isaaclab.sh --format` (the full pre-commit run) was not run; `ruff check` and `ruff format
  --check` at the pinned version (0.14.10) pass on every touched file on both branches.

## Submitted (2026-08-29)

| PR | branch | base | scope |
|---|---|---|---|
| [isaac-sim/IsaacLab#7433](https://github.com/isaac-sim/IsaacLab/pull/7433) | `fix/mimic-max-num-failures` | `develop` | 22 files, 4 commits |
| [isaac-sim/IsaacLab#7434](https://github.com/isaac-sim/IsaacLab/pull/7434) | `fix/cubes-stacked-at-rest` | `develop` | 4 files, 6 commits |

Both carry the contributor line (`* Kai Pei` inserted alphabetically into the project-wide
`CONTRIBUTORS.md`; git resolves the identical insertion on both sides without conflict when the
second one merges). Docker/GPU CI on this repository is on demand: a maintainer or the author has
to comment `run-ci` on each PR for the new tests to execute.
