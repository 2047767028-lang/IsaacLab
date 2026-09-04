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

## Group E — num_envs 10, reseeding, hold at the contact frame (2026-09-02)

Scripts `contact_hold_trial.py` / `build_contact_table.py` / `run_contact_hold.sh` /
`contact_hold_analysis.py`; outputs (hdf5, logs, hit counters, `ref_table.npz`) preserved under
`datasets/arc_sweep_diagnosis_runs/contact_hold/`. Same harness as group D, and the two controls
reproduce group D to the episode (97/300 and 47/300), so the differences below are the interventions.

| run | arc | intervention | result | mechanism check |
|---|---|---|---|---|
| `ref_none` | 0.5 cm | — | **32.3%** | = `d2b_ref` |
| `arc_none` | 3.0 cm | — | **15.7%** | = `d2b_arc` |
| `arc_snap` | 3.0 cm | ramp onto the reference run's achieved contact position over 10 frames, hold it 20 noise-free frames, ramp back | **19.7%** | lookup 290/310, 1229 holds inserted; distance to the reference at contact 1: 0.69 → **0.36 cm** — the retarget worked this time |
| `arc_hold` | 3.0 cm | hold 20 noise-free frames at the **nominal target** before every gripper transition; no reference run | **39.0%** | episodes 230 → 310; reach at grasp 1.42 → **0.85 cm** |
| `ref_hold` | 0.5 cm | same hold | **50.3%** | reach at grasp 1.12 → 0.84 cm; cube offset at release 1.37 → 0.92 cm (source demos: 0.56 / 0.90) |

### Stage funnel (fraction of 300 attempts reaching each stage)

| run | cube_2 lifted | cube_2 on cube_1 | cube_3 lifted | success |
|---|---|---|---|---|
| `ref_none` | 97.7% | 51.7% | 39.0% | 32.3% |
| `ref_hold` | 99.7% | **69.7%** | **63.7%** | **50.3%** |
| `arc_none` | 94.7% | 33.7% | 24.0% | 15.7% |
| `arc_snap` | 95.0% | 42.0% | 27.3% | 19.7% |
| `arc_hold` | 98.7% | 61.3% | 53.0% | 39.0% |

The hold acts on placement and on the second grasp; the first grasp barely fails in any condition.

### Same-scene split by the reference run's own outcome (1 mm layout key, ~half of episodes pair)

| run | scenes where `ref_none` succeeded | scenes where it failed |
|---|---|---|
| `arc_none` | 27.5% | 3.5% |
| `arc_snap` | 47.9% | 6.2% |
| `arc_hold` | **63.8%** | **17.4%** |
| `ref_hold` | 82.4% | 25.3% |

Copying the reference helps only where the reference was right; holding at the nominal target helps
on both kinds of scene, including 17% of the scenes the reference itself lost.

### Why group D's direction-2 runs measured nothing — the second, larger reason

`snap_to_reference` set `delta = ref_contact_pose − last_target_of_subtask`. The subtask's last
target is not the contact frame: `grasp_1`/`grasp_2` fire on the gripper-transition frame itself
(`stack_1` two frames later, measured on all 10 demos), `datagen_info_pool.py:157-163` ends the
subtask one frame after the signal, and `randomize_subtask_boundaries` adds 10–20 more. In the source
demos the arm travels a median **10.0 / 15.1 / 17.3 cm** in the 10 / 15 / 20 frames after a contact
event (`post_contact_motion.py`). The tail was pushed along a 10–17 cm vector that mixed "where the
reference was at contact" with "where the source arm went afterwards". The lag explanation given in
CLAUDE.md 2.13 was at most a minor contributor.

### What this group overturns

- **"Arm position at contact is not the carrier" (REMEDIES.md) is withdrawn.** Remedy 3 dwelled at
  the start of the next subtask and remedy 5 at the subtask end — both 11–23 frames after the gripper
  had acted — so neither converged the arm at contact. The 0.69 cm figure is the median over
  successful grasps; the failing grasps sit at 1.88 cm (`premise_test.py`, row 1). Position at contact
  is the carrier, and the hold that fixes it lifts 3.0 cm arc from 15.7% to 39.0%.
- **"The 5 cm lag is part of the replay and any catch-up pushes the arm where the source never was"
  (CLAUDE.md 2.10) is true in motion and false at contact.** `lag_structure.py`: |target − achieved|
  in the source demos is 4.86 cm median over ordinary frames (91% above 0.3 cm) and **0.00 cm** at the
  gripper-action frames (70% below 0.3 cm). The human releases the stick, the arm settles, then the
  gripper acts (`contact_frames_and_funnel.py` prints one demo frame by frame). At that frame the
  target *is* the source arm's position, so holding at the target reproduces the source.
- **Copying the parallel run's achieved pose is the wrong reference.** That run is 1.12 cm from the
  cube at closure (source human 0.56 cm) and fails 68% of the time; its contact positions carry its
  own noise error and, in most scenes, a failing pose.

### What remains

After the hold, arc 3.0 cm still trails 0.5 cm by 11.3 pp, almost all of it at "cube_2 on cube_1"
(61.3% vs 69.7%): release xy p90 5.34 cm vs 1.70 cm with medians aligned (1.17 vs 0.92). The arm is
in place; the cube in the jaws is not. Cube seat / orientation after the arc-perturbed approach is
the next quantity to measure — the hdf5s here suffice.

### Caveats

One generation seed, n = 300 per run, HOLD = 20 and RAMP = 10 untuned, episodes 35% longer,
`PERTURB_STD` at the task default (0.02) as in group D. Whether a policy trained on held data
behaves differently is untested.

### Upstream note

MimicGen's `SubTaskConfig.num_fixed_steps` is a dwell — but at the *start* of each subtask, and every
shipped Franka stack config sets it to 0. The dwell that matters is before each gripper transition.
Candidate patch: a `num_settle_steps_before_gripper` option (or convergence-gated gripper actions).

## Group F — phase 2/3: second scene draw, approach-phase freeze, production point, convergence gate (2026-09-03)

Runners `run_contact_hold_phase2.sh` / `run_contact_hold_phase3.sh`; same harness as groups D/E;
outputs preserved with group E under `datasets/arc_sweep_diagnosis_runs/contact_hold/`. "draw 2" is
`RESEED_BASE=2000000`, a different fixed scene sequence; everything else is draw 1 and pairable with
group E. Geometry columns are median/p90 in cm (source demos: grasp reach 0.56, release xy 0.90).

| run | arc | freeze | hold before gripper | scenes | result | grasp reach | release xy | len |
|---|---|---|---|---|---|---|---|---|
| `ref_none` (E) | 0.5 cm | 0.3 | — | draw 1 | 32.3% | 1.12 / 2.08 | 1.37 / 4.68 | 230 |
| `ref_hold` (E) | 0.5 cm | 0.3 | fixed 20 | draw 1 | 50.3% | 0.84 / 1.66 | 0.92 / 1.70 | 310 |
| `s2_ref_hold` | 0.5 cm | 0.3 | fixed 20 | **draw 2** | 55.3% | 0.84 / 1.66 | 0.95 / 1.83 | 310 |
| `fz_ref_hold` | 0.5 cm | **0.5** | fixed 20 | draw 1 | 51.3% | 0.84 / 1.63 | 0.90 / 1.62 | 310 |
| `gt_ref` | 0.5 cm | 0.3 | **gated** ≤ 40 | draw 1 | **55.0%** | 0.79 / 1.72 | 0.85 / 1.57 | 302 |
| `arc_none` (E) | 3.0 cm | 0.3 | — | draw 1 | 15.7% | 1.42 / 3.07 | 2.01 / 9.54 | 230 |
| `arc_hold` (E) | 3.0 cm | 0.3 | fixed 20 | draw 1 | 39.0% | 0.85 / 1.99 | 1.17 / 5.34 | 310 |
| `s2_arc_hold` | 3.0 cm | 0.3 | fixed 20 | **draw 2** | 40.3% | 0.89 / 1.86 | 1.16 / 5.15 | 310 |
| `gt_arc` | 3.0 cm | 0.3 | **gated** ≤ 40 | draw 1 | 44.0% | 0.85 / 2.31 | 1.15 / 6.98 | 309 |
| `fz_arc_hold` | 3.0 cm | **0.5** | fixed 20 | draw 1 | **49.7%** | 0.84 / 1.66 | 0.93 / 1.73 | 310 |
| `op_arc_hold` | **1.2 cm** | 0.3 | fixed 20 | draw 1 | **50.0%** | 0.83 / 1.64 | 0.91 / 1.78 | 310 |

Gate statistics (from the hits files): `gt_ref` 1215 gates, mean 17.8 frames, p90 40, 18.8% reached the
40-frame cap without getting inside 0.3 cm; `gt_arc` 1217 gates, mean 19.5, 21.4% at the cap.

### Readings

- **Robust to the scene draw.** The hold's effect reproduces on a second sequence of scenes
  (50.3 → 55.3%, 39.0 → 40.3%).
- **Gating beats a fixed 20-frame hold by ~5 pp at both amplitudes** at the same average length.
  One gate in five hits the cap without converging to 0.3 cm — most plausibly at releases, where the
  held target sits against the cube below and the last millimetres are not reachable; a looser or
  event-specific tolerance has not been tried.
- **The residual arc penalty is an approach-phase effect, and freezing more of the approach removes
  it.** With the arc frozen over the last 50% of each subtask instead of 30%, 3.0 cm arc + hold reaches
  49.7% — indistinguishable from the reference + hold — and the release tail (p90 5.34 → 1.73 cm)
  is gone. The gate alone at freeze 0.3 leaves that tail in place (6.98). Cost: the path-integrated
  perturbation falls by about 30% at the same 3.0 cm peak.
- **Production point.** 1.2 cm + fixed hold 20 gives 50.0%, against 34.1% for the v1 production run
  at 1.2 cm without the hold (visuomotor task) and 33.3% in the v2 sweep.

### Recommended combination on this evidence

Arc on the free segment (freeze 0.5 for peaks up to 3.0 cm, or freeze 0.3 at 1.2 cm) plus a
convergence-gated, noise-free hold before every gripper transition (0.3 cm / ≤ 40 steps). Expected
yield 50–55%, against 32% for stock MimicGen on this task. Not yet run: gate + freeze 0.5 together;
gate at 1.2 cm; a tolerance/cap sweep; a second scene draw for the gate.

### Caveats

n = 300 per cell, one generation seed per draw, episodes 30–35% longer, and no test yet of what a
policy trained on held data does at deployment.

## Group G — amplitude sweep under the gated hold, with the arc the arm actually traces (2026-09-04)

Runners `run_contact_hold_phase4.sh` (0/5/8/10 cm) and `run_contact_hold_phase5.sh` (1.2/3.0 cm
re-run with the commanded-path dump); gate cap 60; freeze 0.3; isotropic directions; scene draw 1;
300 attempts each. Commanded and achieved arcs from `effective_amplitude.py` (distance of the
achieved path to the unperturbed command minus its distance to the perturbed command; the dump is
the trial script's CMD_DIR). Funnel from `stage_funnel.py`.

| nominal | commanded peak | **achieved peak** (median / p90) | achieved / commanded | left at contact | cube_2 on cube_1 | success |
|---|---|---|---|---|---|---|
| 0 cm | 0 | 0 | — | 0.29 cm | 71.1% | **51.5%** |
| 1.2 cm | 1.16 cm | 0.59 / 1.03 | 0.51 | 0.34 | 69.3% | **51.7%** |
| 3.0 cm | 2.79 cm | 1.24 / 2.36 | 0.45 | 0.43 | 61.3% | **41.0%** |
| 5 cm | 4.35 cm | 1.88 / 3.52 | 0.43 | 0.64 | 46.0% | **26.0%** |
| 8 cm | 6.87 cm | 2.86 / 5.32 | 0.42 | 1.36 | 27.0% | **9.7%** |
| 10 cm | 8.34 cm | 3.49 / 6.62 | 0.42 | 1.58 | 16.3% | **6.0%** |

Gate statistics: mean hold 21.7 / 22.2 / 24.6 / 28.4 / 31.4 / 31.9 frames; fraction reaching the
60-step cap 18 / 19 / 22 / 26 / 30 / 30%. Raising the cap from 40 to 60 did not lower the fraction at
0.5 cm (18.8 → 17.9%): those events are blocked, not slow.

### The cause of the 42% (from code and data, `big_arc_cause.py`)

- The command each step is `target − current` (metres) + noise, times `scale = 0.5`
  (`stack_ik_rel_env_cfg.py:39`); the DLS IK turns that into a joint-position target and the PD
  (`FRANKA_PANDA_HIGH_PD_CFG`, stiffness 400 / damping 80) realises part of it in one 50 ms control
  step. Measured on subtask 0 against the perturbed command: achieved step along the error ≈ 0.11 ×
  error for errors under 4 cm, sub-linear above (0.27 → 0.33 → 0.40 cm/frame for 1-2 / 2-4 / 4-8 cm
  errors). **The loop is a first-order tracker with a time constant of about 10 frames.** A bump that
  rises and falls within ~40 frames is attenuated to roughly 40-50% by such a tracker, independent
  of its amplitude — which is what the constant 0.42-0.51 ratio says. Above 5 cm the demanded
  lateral speed (~A/12 cm per frame on top of the path's own 0.5) also runs into the arm's practical
  speed range (achieved steps p90 1.5 cm/frame, the same as in the source demos), hence the slight
  fall in the ratio.
- Not the cause: joint limits (recorded joint ranges are well inside the drive limits; the Isaac
  Lab Franka's joint 4 runs to +2.4 rad, so the URDF table does not apply) and joint-velocity limits
  (3-4% of free-zone frames above 90% of the limit at every amplitude).
- Where the success goes: the first grasp holds up (99.7 → 87.7%); "cube_2 on cube_1" collapses
  (71 → 16%). The cube's seat between the fingers at the first release goes 0.86 → 1.20 → 2.84 cm
  (p90 13.5 cm at 10 cm: the cube has left the jaws). The damage is done during the carry, before
  any hold, by the lateral acceleration the bump demands. A longer hold at the contact frame cannot
  undo it, and the gate's residual at contact (1.4-1.6 cm at 8-10 cm) says the arm is arriving from
  a state the hold cannot fully correct either.

### What follows

The bump is too **fast** for the loop, not too large. Amplitude at fixed duration buys ~0.42 of
itself in achieved deviation and pays in placement. The levers the cause points at, none tested:
(1) stretch the perturbed free zone in time so the bump lasts several time constants (a
`PERTURB_ARC_STRETCH` resampling of the commanded segment); (2) raise the loop gain (`scale` 0.5 →
1.0 was catastrophic without the hold, group A; untested with the gate protecting the contact);
(3) keep the arc small on the carrying subtasks (1 and 3) where the cube can slip. Direction
priors (upper hemisphere / human cone) remain untested; the offline geometry check
(`big_arc_geometry.py`) found table penetration impossible on this task (the arm is 16-21 cm up at
the envelope peak) and isotropic arcs newly grazing an uninvolved cube only 3-4% of the time.

## Group H — slowing the bump: free zone time-stretched 2x (2026-09-04)

Runner `run_contact_hold_phase6.sh`; `PERTURB_ARC_STRETCH=2` repeats every free-zone command frame
twice (frozen tail and contact frame keep their timing); gate cap 60; freeze 0.3; isotropic; draw 1.

| run | arc | stretch | commanded peak | achieved peak (med / p90) | ach / cmd | at contact | cube_2 on cube_1 | success |
|---|---|---|---|---|---|---|---|---|
| `gt_zero` (G) | 0 | 1 | 0 | 0 | — | 0.29 cm | 71.1% | 51.5% |
| `st0` | 0 | 2 | 0 | 0 | — | 0.50 | 68.7% | **52.7%** |
| `big5` (G) | 5 cm | 1 | 4.35 | 1.88 / 3.52 | 0.43 | 0.64 | 46.0% | 26.0% |
| `st5` | 5 cm | 2 | 4.42 | **2.40** / 3.94 | 0.54 | **1.03** | **39.0%** | **22.3%** |
| `big10` (G) | 10 cm | 1 | 8.34 | 3.49 / 6.62 | 0.42 | 1.58 | 16.3% | 6.0% |
| `st10` | 10 cm | 2 | 8.36 | **4.57** / 7.69 | 0.55 | **2.83** | **10.3%** | **4.3%** |

Direction dependence (`arc_direction_dependence.py`): the achieved fraction barely depends on the
arc's angle to the base–eef axis (0.33–0.43), so this is not a singular posture; it depends on the
vertical component (horizontal arcs 0.44–0.45, vertical ones 0.25–0.28 at 5–10 cm) and on the
subtask (carries 0.44–0.46, the cube_3 approach 0.28).

### Readings

- Slowing the free zone alone changes nothing (0 cm: 51.5 → 52.7%), so it is a clean control.
- The stretch does what the low-pass model says: the arm follows more of the bump (0.38 → 0.59
  overall, in line with a ~10-frame tracker on an 80-frame bump). **Following the bump better
  makes placement worse.** The arm enters the frozen tail farther from the nominal path
  (2.4 vs 1.9 cm at 5 cm; 4.6 vs 3.5 at 10 cm), the 18-frame tail plus the gate do not bring it
  back (residual at contact 0.64 → 1.03 cm, 1.58 → 2.83 cm), and the cube's seat at release
  worsens (1.20 → 1.75 cm at 5 cm).
- So above ~3 cm the limiting factor is not "the bump is too fast to follow" but "a followed bump
  cannot be undone in the frozen tail, and the carry disturbs the cube on the way". Time at the
  contact frame does not help (group G: raising the gate cap changed nothing); time in the free
  zone makes it worse.

### What this leaves

The losses at every large amplitude sit in the carrying subtasks (1 and 3: cube_2 on cube_1
collapses first). The approach subtasks (0 and 2, empty gripper) are where a different path is
wanted for training and where nothing is in the jaws to disturb. Untested: arc on the approach
subtasks only, at 5–10 cm, under the gate (group A had `ARC_SUBTASKS=0` at 3 cm without any hold:
33.3% against 22.0% for all four). Also untested: a horizontal-only direction prior (vertical arcs
are followed worst and return worst), and a longer frozen tail for large arcs (freeze 0.5 fixed
3 cm in group F at ~30% less integrated perturbation).

### Group H addendum — the seat is born at the grasp, not in the carry (`seat_grasp_vs_release.py`)

| run | seat at grasp | seat at release | carry slip (release − grasp) | carry accel max | accel p90 |
|---|---|---|---|---|---|
| source demos | 0.32 / 0.82 | 0.51 / 0.80 | 0.06 / 0.46 | 0.66 / 0.94 | 0.36 / 0.46 |
| `gt_zero` | 0.61 / 1.20 | 0.86 / 1.54 | 0.20 / 0.93 | 1.34 / 1.77 | 0.86 / 1.00 |
| `gt30` | 0.73 / 1.65 | 0.92 / 2.00 | 0.21 / 1.08 | 1.35 / 1.68 | 0.83 / 0.96 |
| `big5` | 0.85 / 2.91 | 1.20 / 4.33 | 0.29 / 2.87 | 1.33 / 1.73 | 0.81 / 0.96 |
| `st5` | 1.18 / 4.10 | 1.75 / 4.76 | 0.26 / 1.80 | 1.42 / 1.87 | 0.88 / 0.96 |
| `big10` | 1.85 / 5.09 | 2.84 / 13.45 | 0.38 / 9.58 | 1.35 / 1.77 | 0.80 / 0.92 |
| `st10` | 2.49 / 7.30 | 3.41 / 8.79 | 0.19 / 3.26 | 1.38 / 1.73 | 0.85 / 0.96 |

(cm and cm/frame², median / p90.) Carry accelerations do not change with the arc at all — the
tracker smooths the bump and the action noise sets them (at twice the source demos'). The offset
that breaks placement is already there at closure and grows with amplitude; the carry adds
0.2–0.4 cm at every amplitude, with a real tail only at 10 cm (cube dropped or rotated in hand).
**Withdrawn**: "the carry slips the cube when the arc exceeds the noise" (the user's objection —
slip is an inertial-force condition — stands, and it is not reached), and the "arc on the grasp
segments only" suggestion (the carry segments tolerate the arc; the grasp segments' approach is
where the damage originates).

Terminology fixed by the user (2026-09-04): **抓取段** = subtasks 0/2 (empty gripper approaching a
cube); **搬运段** = subtasks 1/3 (carrying); **冻结段** = the frozen target at the end of every
subtask where the arm is let approach; **接近段** = the part of any subtask where the gripper
approaches the object it interacts with. Design direction: keep the arc on the whole trajectory;
make the bump's return complete before the 冻结段 (peak earlier via the unused v3
`PERTURB_ARC_PEAK_FRAC`, freeze length growing with amplitude) so the 接近段 carries source-like
lag and direction; acceptance metric = seat at grasp back near the 0 cm run's 0.61 cm.

## Group I — end the bump's return before the frozen segment: early peak, longer freeze (2026-09-04)

Runner `run_contact_hold_phase7.sh`. Arc on the whole trajectory, peak amplitude untouched; the
envelope peak moved from 50% to 25% of the free zone (`PERTURB_ARC_PEAK_FRAC_MIN/MAX=0.25`, the v3
knob's first use; dump check: peak at 17% of the segment, expected 17.5%), freeze 0.3 or 0.5;
gate cap 60; isotropic; scene draw 1; 300 attempts each. Controls are group G's `big5` / `big10`
(peak 0.5, freeze 0.3) and `gt_zero`.

| run | arc | peak | freeze | seat at grasp (med / p90) | left at contact | achieved arc peak (med / p90) | cube_2 on cube_1 | success |
|---|---|---|---|---|---|---|---|---|
| `gt_zero` | 0 | — | 0.3 | 0.61 / 1.20 cm | 0.29 cm | 0 | 71.1% | **51.5%** |
| `big5` | 5 cm | 0.5 | 0.3 | 0.85 / 2.91 | 0.64 | 1.88 / 3.52 | 46.0% | 26.0% |
| `pk5` | 5 cm | **0.25** | 0.3 | **0.69 / 1.48** | 0.35 | 1.78 / 3.78 | 69.7% | **48.0%** |
| `pk5f` | 5 cm | **0.25** | **0.5** | 0.68 / 1.29 | 0.31 | 1.47 / 3.40 | 71.7% | **51.0%** |
| `big10` | 10 cm | 0.5 | 0.3 | 1.85 / 5.09 | 1.58 | 3.49 / 6.62 | 16.3% | 6.0% |
| `pk10f` | 10 cm | **0.25** | **0.5** | **0.68 / 1.42** | 0.33 | **2.62 / 6.03** | 67.0% | **46.7%** |

Carry slip and carry accelerations unchanged (0.16–0.23 cm; 1.34–1.36 cm/frame² max) — as in the
group H addendum, the carry was never the problem. Gate statistics back to the 0 cm run's
(mean hold 22.3–22.8 frames, 18–19% at the cap, against 28–32 frames and 26–30% for `big5`/`big10`).

### Reading

The acceptance metric lands: the seat at grasp returns to 0.68–0.69 cm (0 cm run 0.61) at both
5 and 10 cm, the residual at contact to 0.31–0.35 cm (0.29), and the funnel to within 4 pp of the
0 cm run at every stage. **Same achieved arc, yield restored**: `pk5` keeps 1.78 cm of achieved
bump (against `big5`'s 1.88) at 48.0% instead of 26.0%; `pk10f` keeps 2.62 cm median / 6.0 cm p90
at 46.7% instead of 6.0%. What changed is only *when* the command's return ends: at peak 0.25 the
return spans ~75% of the free zone (≈30 frames at freeze 0.3), several tracker time constants,
so the arm re-joins the nominal path before the frozen segment and enters the approach segment
with source-like lag and direction. This is the user's principle of 2026-09-04 confirmed:
perturb the whole trajectory, do not chase the perturbed target, converge before the frozen
segment, freeze and settle at the end.

### Not yet done

Second scene draw; 8 cm and the peak position as a sweep (0.15 / 0.25 / 0.35); freeze 0.3 at
10 cm (only 0.5 was run); the "stopped" gate criterion; direction priors; the training-side test.
