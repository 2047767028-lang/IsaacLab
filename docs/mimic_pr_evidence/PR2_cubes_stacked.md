# Description

`cubes_stacked` decides the stack tasks' success from an instantaneous geometric configuration — each cube within `xy_threshold` of the one below it and one cube-height above it — with no requirement that the cubes be supported or stationary. A cube released above its target satisfies exactly that description for a frame or two on the way down.

That matters more than it sounds, because the Mimic data generator accumulates success across the whole episode:

```python
generated_success = generated_success or exec_success
```

so a single frame of a cube passing through the right height marks the entire episode a success, and it is written into the generated dataset as a demonstration of the task being completed.

### Evidence

Replaying the criterion offline over two generated datasets (the reproduction was validated first by confirming it passes 100% of the demos the generator accepted):

| dataset | accepted demos | stack broken at the final frame |
|---|---|---|
| A | 380 | **28 (7.4%)** |
| B | 358 | **19 (5.3%)** |

In those demos the top cube ends a median 6.3 cm away in xy and at the *same height* as the cube it should be sitting on — it is on the table, not on the stack. The discriminator is what the mechanism predicts: cube_3's speed at the qualifying frame is a median 0.028–0.032 m/s in the broken demos versus **0.0000 m/s** in the rest.

One timeline, taken from the case most favourable to any other explanation — the one whose jaws came closest to fully open while the criterion held:

```
frame  stacked  cube3-cube2 dz  cube3 z   jaw error
  191            7.91 cm        14.62 cm   18.465 mm   <- held above the target
  193            5.60 cm        12.33 cm    3.124 mm   <- jaws opening, cube falling
  195     *      5.01 cm        11.76 cm    0.424 mm   <- criterion fires, mid-fall
  196     *      4.30 cm        11.04 cm    0.135 mm
  199            0.33 cm         4.59 cm    0.012 mm   <- cube reaches the table
  205            0.00 cm         2.05 cm    0.007 mm   <- 6.8 cm from cube_2
```

The gripper releases 7.5 cm above `cube_2`; the cube free-falls through the height the criterion is looking for and lands on the table.

### The fix

Require the cubes to have settled. `max_lin_vel` defaults to `0.01` m/s — comfortably above the noise floor of a cube at rest and well below anything falling — and can be set to `None` to skip the check.

The default is not guessed. Replaying 100 generated demos through the recorded `root_velocity` — the same signal the check reads — shows cubes resting on the stack are **not** at zero: the contact solver leaves a median 0.0132 m/s, peaking at 0.0341 m/s over the 91 sound demos, against a median 0.104 m/s for a cube caught mid-fall.

| max_lin_vel | keeps sound demos | keeps defective ones |
|---|---|---|
| 0.010 | 15/91 | 0/9 |
| 0.020 | 77/91 | 0/9 |
| **0.050** | **91/91** | **2/9** |
| 0.100 | 91/91 | 4/9 |

It is deliberately not a perfect separator: the distributions overlap at the bottom because a cube near the apex of a bounce is briefly slow, and tightening past 0.034 m/s starts discarding real demonstrations. An angular-velocity term buys one more defect at the cost of four sound demos, so it was left out.

Measured end to end on the stock `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0`, generating 100 accepted demos each way:

| criterion | accepted | ends broken | attempts | success rate |
|---|---|---|---|---|
| before | 100 | **9 (9.0%)** | 279 | 35.8% |
| after | 100 | **2 (2.0%)** | 300 | 33.3% |

### A note on the gripper tolerance, which looks like the opposite bug

`atol`/`rtol` of `1e-4` against `gripper_open_val` work out to 0.104 mm on a 40 mm jaw travel, and 11% of failed attempts had reached the stacked geometry but missed that tolerance — which reads like a false-negative worth relaxing. It is not. Every one of those 76 episodes is a drop: none has its stack intact at the final frame, and the top cube has moved a median 10.8 cm by then. Relaxing the tolerance to 3 mm would have admitted 54 of them into the dataset. The tight tolerance is masking this defect by accident; it is not causing one, and it is left alone here.

## Type of change

- Bug fix (non-breaking change which fixes an issue)

This lowers reported success rates for the stack tasks slightly, in both Mimic generation and policy evaluation, because episodes that end with the cube on the table stop counting. `max_lin_vel=None` in the termination term's `params` restores the previous behaviour.

## Checklist

- [x] I have read and understood the contribution guidelines
- [ ] I have run the `pre-commit` checks with `./isaaclab.sh --format`
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [x] I have added a changelog fragment under `source/<pkg>/changelog.d/` for every touched package
- [ ] I have added my name to the `CONTRIBUTORS.md` or my name already exists there
