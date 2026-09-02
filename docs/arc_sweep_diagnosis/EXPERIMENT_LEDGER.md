# Experiment ledger — every generation run, and which ones may be compared

Twenty-six runs across six batches. They are **not** mutually comparable: two harness settings move
the absolute success rate more than most of the interventions do.

## Read this first: the comparability trap

The same nominal configuration — arc 3.0 cm, no intervention — returned three different numbers:

| run | num_envs | reseeding | result |
|---|---|---|---|
| `s1_ref_high` | 1 | no | **28.0%** |
| `ref_high` | 10 | no | **22.0%** |
| `d2b_arc` | 10 | yes | **15.7%** |

A 12.3 pp spread from harness settings alone, against interventions whose real effects are 3–10 pp.
**Only compare runs inside the same group below.** Comparing `gate1_high` (33.3%, group A) against
`d2b_arc` (15.7%, group D) would suggest a +17.6 pp effect that is mostly the harness.

`num_envs` matters because the startup `env.reset()` draws every environment's scene from one seed,
so a 1-env and a 10-env run see different scene sets from the first episode. Reseeding matters
because it replaces the free-running RNG with a fixed per-episode seed, which changes both the scene
draw and the action-noise stream.

## Group A — num_envs 10, no reseeding

The main remedy sweep. Reference points reproduce the v2 arc sweep (0.5 cm: 38.0% here vs 35.1%
there; 3.0 cm: 22.0% vs 19.1%), so this group is anchored to the earlier work.

| run | arc | intervention | result | note |
|---|---|---|---|---|
| `ref_low` | 0.5 cm | — | **38.0%** | group reference |
| `ref_high` | 3.0 cm | — | **22.0%** | group reference |
| `dwell_low` | 0.5 cm | `num_fixed_steps=20` | 42.3% | lifts both ends, gap unchanged |
| `dwell_high` | 3.0 cm | `num_fixed_steps=20` | 24.7% | ← |
| `scale_low` | 0.5 cm | `arm_action.scale=1.0` | 22.0% | catastrophic at both ends |
| `scale_high` | 3.0 cm | `arm_action.scale=1.0` | 15.7% | ← |
| `tail_low` | 0.5 cm | end dwell, **noise inherited** | 37.0% | **INVALID**, see below |
| `tail_high` | 3.0 cm | end dwell, **noise inherited** | 20.0% | **INVALID** |
| `tail2_low` | 0.5 cm | end dwell, noise-free | 35.0% | converges best, success unchanged |
| `tail2_high` | 3.0 cm | end dwell, noise-free | 19.7% | ← |
| `gate1_high` | 3.0 cm | arc on subtask 0 only | **33.3%** | works; dominated by lowering amplitude |
| `gate2_high` | 3.0 cm | arc on subtasks 0, 2 | 28.3% | ← |

## Group B — num_envs 1, no reseeding

Run to test whether a single environment made episodes pairable. It did not.

| run | arc | intervention | result |
|---|---|---|---|
| `s1_ref_low` | 0.5 cm | — | **40.7%** |
| `s1_ref_high` | 3.0 cm | — | **28.0%** |
| `s1_dwl_low` | 0.5 cm | end dwell, noise-free | 32.7% |
| `s1_dwl_high` | 3.0 cm | end dwell, noise-free | 21.3% |

The end dwell costs 8.0 pp and 6.7 pp here — a much clearer signal than group A gave, and the
clearest evidence that helping the arm catch up to its target actively hurts.

## Group C — num_envs 1, reseeding

| run | arc | intervention | result | note |
|---|---|---|---|---|
| `d2_ref` | 0.5 cm | recording boundary poses | 35.8% | the atexit dump never fired |
| `d2_arc` | 3.0 cm | — | 22.5% | |
| `d2_snap` | 3.0 cm | snap to reference | **INCOMPLETE** | crashed: reference file missing |

## Group D — num_envs 10, reseeding

| run | arc | intervention | result | note |
|---|---|---|---|---|
| `d2b_ref` | 0.5 cm | — | **32.3%** | group reference; source of the pose table |
| `d2b_arc` | 3.0 cm | — | **15.7%** | group reference |
| `d2b_snap` | 3.0 cm | snap, tolerance 1e-4 | 15.7% | **INVALID**, lookup missed |
| `d2c_snap` | 3.0 cm | snap, tolerance 2e-3 | 15.7% | **INVALID**, wrong frame |
| `d2c_nodwell` | 3.0 cm | snap without hold | 15.7% | **INVALID**, wrong frame |
| `f_snap_hold` | 3.0 cm | snap + hold, frame fixed | **12.0%** | valid: lookup 1137/92, episodes 239→319 |
| `f_snap_only` | 3.0 cm | snap without hold | **13.0%** | valid: lookup 1147/80 |
| `f_hold_only` | 3.0 cm | hold at the nominal target | **17.3%** | |

### Direction 2 moved the gripper the wrong way

Distance from each run's own contact poses to the reference run's achieved poses at the same four
events, matched by layout:

| run | contact 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| `d2b_arc` (no intervention) | **0.69 cm** | 0.90 | 0.99 | 3.79 |
| `f_snap_only` | 0.85 | 1.01 | 2.40 | 9.86 |
| `f_snap_hold` | **1.59 cm** | 1.97 | 3.16 | 12.18 |
| `f_hold_only` | 1.39 | 1.94 | 2.44 | 10.13 |

The uncorrected run is already the closest. Retargeting set the *target* to the reference's
*achieved* pose, and since the arm trails a target by about 5 cm it landed that far short -- whereas
leaving the target alone lets it trail the same target the reference trailed, arriving where the
reference arrived. **Direction 2 has therefore not had a fair test**; the correct form adds the
difference between the two runs' lags, which is a per-step feedback quantity, not a snap.

What the run does establish, more directly than anything before it: the uncorrected arm is within
**0.69 cm** of the reference at the first contact, against 1.66 cm of grasp clearance per side, and
success still falls 32.3% → 15.7%.

## Runs that measured nothing

Five runs produced numbers that look like results and are not. Each was caught by a mechanism check
rather than by the success rate, which in every case looked perfectly plausible.

| run(s) | what went wrong | how it showed |
|---|---|---|
| `tail_low`, `tail_high` | appended dwell frames inherited `action_noise=0.03`, 3 cm per axis, so the arm was shoved around faster than it could settle | frames within 1 cm of target: 0.9% against 0.8% with no dwell at all, where the shipped dwell reached 2.8% |
| `d2b_snap` | scene match required 1e-4 m; reseeding reproduces a layout to about 0.2 mm median, 1.4 mm p90 | 44% of episodes found no reference |
| `d2c_snap`, `d2c_nodwell` | runtime lookup compared world coordinates against an env-local table, so nine environments in ten could never match | control and both corrected runs returned an identical 47/300, and episode length stayed at 239 |

The tell in the last case is worth keeping: **three runs returning exactly the same count is not a
null result, it is a broken intervention.** A real trajectory change cannot land on the same integer.

## Standing procedure

Before a full run of any intervention, check one metric that says *it happened*:

- appended dwell → episode length grew (239 → 319 for a 20-frame dwell over four subtasks)
- reference lookup → hits/misses written to `snap_hits.txt` (172/0 once the frame bug was fixed)
- convergence claims → p05 of `|target − current|`, not the success rate

And never gate a queue on `pgrep -f <pattern>`: twice a queue hung for hours because the shell that
launched it, or the one that checked on it, carried the pattern in its own command line. Tightening
the pattern does not fix this; the next command that mentions the new pattern hangs it again.
