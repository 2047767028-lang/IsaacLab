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
