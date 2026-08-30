# Why arc generation success falls with amplitude

## The claim under test

> The `sin²`-family envelope is zero in value *and* slope at both ends of the free zone, and the
> trailing `freeze_frac` of every subtask is byte-identical to the source. So the contact phase
> cannot be affected, and a success rate that still collapses must mean the criterion is treating
> the arc group unfairly.

The premise is true of the **target** pose sequence — `_apply_arc_perturbation` writes only
`new_poses[:free_len]` — and false of the **achieved** state. The arm tracks those targets through a
differential IK controller, and a 7-DOF arm reaching the same end-effector pose along a different
path does not have to arrive in the same joint configuration.

## What the curve actually looks like

| cm | 0.5 | 0.8 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|---|
| success | 35.1% | 34.4% | 33.1% | 30.1% | 27.8% | 23.9% | 19.1% |

Not a cliff. A smooth, monotone, convex decline that begins around 0.9 cm: about −5.0 pp/cm over
0.5→1.5, about −9.6 pp/cm over 2.5→3.0. A criterion artifact that switches on at a threshold would
show a knee; this does not have one.

## The paired measurement

The sweep's per-run seed produces the **same scene sequence at every amplitude** — all 500 initial
cube layouts at 3.0 cm match a layout at 0.5 cm to <1e-6 m (`pair_check.py`). So attempt *k* at one
amplitude and attempt *k* at another start from identical cube placements, and scene difficulty
drops out of the comparison entirely.

Restricting to scenes that **succeed at both** amplitudes, and measuring the achieved end-effector
deviation over the last 15 frames — which sit inside the final subtask's frozen tail, where the two
runs are commanded byte-identically:

| cm | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| 0.6 | 0.182 cm | 0.175 cm | 0.181 cm |
| 1.0 | 0.425 cm | 0.458 cm | 0.416 cm |
| 1.5 | 0.651 cm | 0.642 cm | 0.658 cm |
| 2.0 | 0.897 cm | 0.921 cm | 1.039 cm |
| 2.5 | 1.130 cm | 1.019 cm | 1.206 cm |
| 3.0 | **1.383 cm** | **1.295 cm** | **1.351 cm** |

Roughly 45% of the arc amplitude survives into the frozen tail, at every amplitude and on every
seed. The identical target does not produce an identical state.

Paired placement error moves with it (seed 1, same-scene, both-succeed): the cube_1→cube_2 offset
grows from −0.005 cm at 0.6 cm amplitude to **+0.385 cm** at 3.0 cm, and the share of scenes where
the arc run places *worse* than its own 0.5 cm counterpart rises from 49% (chance) to **72%**.

## Lag or null-space drift?

`nullspace_check.py`, seed 1, by distance from the end of the episode:

| cm | eef @−30 | eef @−20 | eef @−10 | eef @−1 | joints @−30 | joints @−1 |
|---|---|---|---|---|---|---|
| 1.0 | 0.501 cm | 0.405 cm | 0.381 cm | 0.360 cm | 0.0433 rad | 0.0441 rad |
| 2.0 | 1.239 cm | 0.871 cm | 0.794 cm | 0.772 cm | 0.0857 rad | 0.0850 rad |
| 3.0 | 2.084 cm | 1.373 cm | 1.244 cm | 1.187 cm | 0.1227 rad | 0.1051 rad |

Frame −30 is still in the free zone, −20 onward is inside the frozen tail. The end-effector error
drops on entering the tail and then **plateaus**; the joint configuration barely moves at all — about
0.105 rad (~6°) apart at 3.0 cm, and just as far apart on the final frame as thirty frames earlier.

The arm has settled into a different configuration rather than lagging on its way back to the old
one. A longer `freeze_frac` would not recover this: the residual has stopped decaying well before
the episode ends.

## Was the criterion unfair to the arc group?

No — it runs the other way.

**The criterion-attributable failure mode shrinks with amplitude.** "Geometry satisfied at some
frame but the gripper-open check never passed" is the one failure that is the criterion's doing
rather than the physics':

| seed | share at 0.5 cm | share at 3.0 cm | absolute count |
|---|---|---|---|
| 1 | 8.6% | 6.9% | 28 → 28 |
| 4 | 11.5% | 7.6% | 37 → 31 |
| 5 | 12.5% | 7.7% | 40 → 31 |

Flat or falling, in share and in absolute count. If the criterion were penalising the arc group, this
is the column that would grow.

**The lost successes are grasp and placement failures.** Same seeds, converting shares to counts out
of 500 attempts (seed 1, 0.5 cm → 3.0 cm): "cube_2 never moved" 2 → 28 (the arm failed to pick up the
first cube at all — a free-space failure with nothing to do with any criterion), "cube_2 grasped but
not stacked" 191 → 250. Those two account for the entire increase in failures; the cube_3 modes and
the criterion mode do not grow.

**The one real criterion bug helps the arc group.** `cubes_stacked` fires on an instantaneous
configuration, so a cube passing through the stacked pose mid-fall counts (this is what PR #7434
fixes). Among accepted demos, the share that end with the stack broken rises with amplitude
(5.1% → 7.5% on seed 1, peaking 11.9%). The stock criterion is *generous* to arc, and the at-rest fix
will lower the arc numbers more than the baseline ones.

## What this changes

The design document's central mechanism — anchor the contact phase, perturb only free space — does
not hold as stated. Freezing the target for the trailing 30% of a subtask does not freeze the state
the arm arrives in. About 45% of whatever is injected in free space is still present at contact, it
scales linearly with amplitude, and because success is a hard threshold on placement, a linearly
growing placement error produces exactly the convex success curve that was observed.

That also explains why the v2 sweep never found a plateau: the quantity driving failure grows
linearly and without bound, so there is nothing to plateau against.

## Scripts

| script | what it establishes |
|---|---|
| `pair_check.py` | scene sequences are identical across amplitudes, so episodes can be paired |
| `paired_contact.py` | frozen-tail end-effector deviation and paired placement error |
| `nullspace_check.py` | the residual is a settled joint-configuration difference, not decaying lag |
| `arc_sweep_diagnosis.py` | placement accuracy among successes, failure-mode split, in-flight loophole |
