# Can the arc penalty be engineered away?

Twelve generation runs, 300 fixed attempts each, on `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-Perturbed-v0`
at seed 1. The reference points reproduce the v2 sweep (38.0% at 0.5 cm against the sweep's 35.1%,
22.0% at 3.0 cm against 19.1%), so the harness is measuring the same thing.

## Results

| intervention | 3.0 cm | 0.5 cm | gap | verdict |
|---|---|---|---|---|
| none | 22.0% | 38.0% | 16.0 pp | — |
| controller gain 0.5 → 1.0 | 15.7% | **22.0%** | 6.3 pp | **worse at both**; the narrower gap is the reference collapsing, not the penalty lifting |
| dwell 20 at subtask **start** (`num_fixed_steps`) | 24.7% | 42.3% | 17.6 pp | lifts both by 3–4 pp, gap unchanged; and episodes are 35% longer, so demos per unit time fall 18% |
| dwell 20 at subtask **end**, noise-free | 19.7% | 35.0% | 15.3 pp | no effect on success despite converging the arm 5× better |
| arc on subtask 0 only | 33.3% | — | — | works, but dominated (below) |
| arc on subtasks 0 and 2 | 28.3% | — | — | same |

## What each one told us

**Raising the gain.** The action is `target − current` and `scale` is the fraction of that gap
commanded per step. Raising it does reduce the tracking lag (7.39 → 6.38 cm) but wrecks placement
(154 → 216 failures): a faster arm flicks the cube as it lets go. Reducing the lag is not
automatically good — how you reduce it decides whether it helps.

**Dwell at the subtask start.** The shipped `num_fixed_steps`, never used by any config. It works
mechanically — frames within a centimetre of target go from 0.8% to 2.8% — and it lifts generation
yield by 3–4 pp at every amplitude. That is a real if modest finding about MimicGen generally. It
does not touch the arc penalty: the gap is 17.6 pp against 16.0 pp.

**Dwell at the subtask end.** Not exposed by any config; implemented by repeating the subtask's last
pose. The first attempt was invalid — the appended frames inherited `action_noise = 0.03`, i.e. 3 cm
per axis, so the arm was shoved around faster than it could settle (0.9% of frames within a
centimetre, against 0.8% with no dwell at all). With the noise zeroed, as the shipped fixed segment
does, convergence became the best of any run: p05 of `|target − current|` fell from 2.50 cm to
**0.46 cm** and 9.4% of frames landed within a centimetre.

**Success did not improve.** 19.7% against 22.0%. That is the most informative result here: the arm
being in the right place at the moment of contact is not what decides the outcome.

**Gating to fewer subtasks.** Excess failures over the low-amplitude reference run 14, 29 and 48 for
k = 1, 2, 4 perturbed subtasks — linear in k (14, 28, 56), not the sqrt(k) an accumulating drift
would give. The damage is local: perturbing only subtask 0 leaves placement failures at 115 against
the reference's 114, and raises only the grasp it actually touches (15 against 4).

This corrects the accumulation story. The residual does accumulate across subtasks -- 0.63, 0.96,
1.01, 1.28 cm at the four contact events, a good sqrt(k) fit -- but that accumulated residual is not
what causes the failures. Correlation was mistaken for causation.

Gating is also dominated. At equal success, `k=1 at 3.0 cm` (33.3%) and `k=4 at 1.5 cm` (32.6%
rescaled) inject amplitude×k of 3.0 against 6.0. Perturbing everything at half amplitude gives twice
the perturbation for the same cost, so gating is a worse point on the same trade-off, not a new one.

## What carries the damage

Three candidates measured and eliminated:

| candidate | test | result |
|---|---|---|
| joint configuration | stratify grasp outcome by gripper-position error | d collapses to −0.65 / +0.24 inside narrow bands; a proxy for gripper position |
| arm position at contact | noise-free end dwell converges it 5× better | success unchanged |
| cube's seat in the jaws | lateral offset at grasp vs placement outcome | d = 0.22–0.32, correlation with placement error 0.02–0.18 |

**What does carry it is not established.** The honest position is that the damage is local to each
perturbed subtask and scales with the amplitude applied there, and that none of the three mechanisms
above transmits it.

## Practical conclusion

None of the cheap interventions removes the arc penalty. The one lever that works is the amplitude
itself, and the earlier operating point of 1.2 cm sits about where the grasp clearance budget says it
should: the cube is 4.68 cm in jaws that open to 8 cm, leaving 1.66 cm of clearance per side, and the
contact-phase residual is about 45% of the amplitude — 0.54 cm at 1.2 cm, a third of the budget,
against 1.35 cm at 3.0 cm, four fifths of it.

## Two process notes

Both cost a run and are worth remembering.

- The appended dwell inherited the subtask's action noise, so the first end-dwell test measured a
  noisy dwell rather than a dwell. Caught by checking a mechanism metric (does the arm actually
  converge?) rather than trusting the success rate alone.
- Two queued scripts both waited on "no `fix_trial.py` running", which was true for both at the
  instant the first batch ended, so two Isaac Sim instances launched onto the same 12 GB card. Queue
  on the other queue's completion marker, not on a shared idle condition.


## Superseded (2026-09-02, group E)

The "what carries the damage" table above is withdrawn. Neither dwell converged the arm at the
contact frame (both sit 11–23 frames after the gripper transition), so "arm position at contact"
was never tested here. A 20-frame noise-free hold placed before each gripper transition lifts
3.0 cm arc from 15.7% to 39.0% and the 0.5 cm reference from 32.3% to 50.3%. See
EXPERIMENT_LEDGER.md group E and `contact_hold_trial.py`.
