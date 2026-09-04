"""At the two contact events that decide the task (first grasp, first release): where is the arm
relative to its target and to the cube -- in the SOURCE demos versus in GENERATED episodes?

Source demos: |target - achieved| == |actions[:, :3]| by construction (action_to_target_eef_pose
adds the unscaled delta to the current pose). Generated episodes: the recorded action also carries
the injected noise, so instead of the action we use the geometric quantity that matters, the
gripper-to-cube distance, and compare it against the source's value at the same event.

Events are found from the data: grasp = first frame the jaws leave the open value by >1 mm;
release = first frame after that where the jaws re-open by >1 mm from their grasped value.
"""
import h5py
import numpy as np

SRC = "/home/pk/IsaacLab/datasets/annotated_dataset.hdf5"
GEN = "/home/pk/IsaacLab/datasets/perturbation_sweep_v2_arc"
OPEN_VAL = 0.04
CUBES = ("cube_1", "cube_2", "cube_3")


def jaw_of(grip):
    return np.minimum(grip[:, 0], -grip[:, 1])


def events(grip):
    jaw = jaw_of(grip)
    closed = np.where(jaw < OPEN_VAL - 0.001)[0]
    if not len(closed):
        return None, None
    g = int(closed[0])
    grasped = jaw[g + 5] if g + 5 < len(jaw) else jaw[g]
    reopen = np.where(jaw[g + 5:] > grasped + 0.001)[0]
    r = int(g + 5 + reopen[0]) if len(reopen) else None
    return g, r


def pause_before(lag, t, thresh=0.003):
    """Consecutive frames before t with |target-achieved| < thresh."""
    n = 0
    i = t - 1
    while i >= 0 and lag[i] < thresh:
        n += 1
        i -= 1
    return n


def q(v):
    v = np.asarray(v, float) * 100
    return f"median {np.median(v):5.2f}  p90 {np.percentile(v, 90):5.2f} cm (n={len(v)})"


print("=" * 78)
print("SOURCE DEMOS (annotated_dataset.hdf5)")
print("=" * 78)
src = {"lag_g": [], "lag_r": [], "pause_g": [], "pause_r": [], "reach_g": [], "reach_r_xy": [], "reach_r_z": []}
with h5py.File(SRC, "r") as f:
    for k in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])):
        d = f["data"][k]
        act = d["actions"][:]
        pos = d["obs/eef_pos"][:]
        grip = d["obs/gripper_pos"][:]
        c1 = d["states/rigid_object/cube_1/root_pose"][:, :3]
        c2 = d["states/rigid_object/cube_2/root_pose"][:, :3]
        lag = np.linalg.norm(act[:, :3], axis=1)
        g, r = events(grip)
        src["lag_g"].append(lag[g]); src["pause_g"].append(pause_before(lag, g))
        src["reach_g"].append(np.linalg.norm(pos[g] - c2[g]))
        if r is not None:
            src["lag_r"].append(lag[r]); src["pause_r"].append(pause_before(lag, r))
            src["reach_r_xy"].append(np.linalg.norm((pos[r] - c1[r])[:2]))
            src["reach_r_z"].append((pos[r] - c1[r])[2])
        print(f"  {k}: grasp@{g} lag {lag[g]*100:.2f}cm pause {pause_before(lag, g)} frames | "
              f"release@{r} lag {lag[r]*100:.2f}cm pause {pause_before(lag, r)} frames | "
              f"gripper action values {sorted(set(np.round(act[:, -1], 2)))}")
print()
print("  at GRASP   |target-achieved|:", q(src["lag_g"]))
print("             frames of near-zero command before it: median", np.median(src["pause_g"]), "min", min(src["pause_g"]))
print("             gripper-to-cube_2 distance:", q(src["reach_g"]))
print("  at RELEASE |target-achieved|:", q(src["lag_r"]))
print("             frames of near-zero command before it: median", np.median(src["pause_r"]), "min", min(src["pause_r"]))
print("             gripper-to-cube_1 xy offset:", q(src["reach_r_xy"]), " z:", q(src["reach_r_z"]))


def key_of(ro):
    return tuple(np.round(np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]), 6))


def load_run(seed, cm):
    tag = f"cm{cm:.1f}".replace(".", "p")
    out = {}
    for suffix, ok in (("", True), ("_failed", False)):
        with h5py.File(f"{GEN}/seed{seed}/generated_{tag}{suffix}.hdf5", "r") as f:
            d = f["data"]
            for k in d.keys():
                obs = d[k]["obs"]
                ro = d[k]["states"]["rigid_object"]
                out[key_of(ro)] = {
                    "success": ok,
                    "pos": obs["eef_pos"][:],
                    "grip": obs["gripper_pos"][:],
                    "c1": ro["cube_1"]["root_pose"][:, :3],
                    "c2": ro["cube_2"]["root_pose"][:, :3],
                }
    return out


print()
print("=" * 78)
print("GENERATED (seed 1): same events, gripper-to-cube geometry")
print("=" * 78)
ref = load_run(1, 0.5)
arc = load_run(1, 3.0)
for name, run in (("arc 0.5cm (reference)", ref), ("arc 3.0cm", arc)):
    reach_g, xy_r, z_r, placed = [], [], [], []
    for ep in run.values():
        g, r = events(ep["grip"])
        if g is None:
            continue
        reach_g.append(np.linalg.norm(ep["pos"][g] - ep["c2"][g]))
        if r is not None:
            xy_r.append(np.linalg.norm((ep["pos"][r] - ep["c1"][r])[:2]))
            z_r.append((ep["pos"][r] - ep["c1"][r])[2])
            # did cube_2 end up on cube_1? (xy within 2 cm at the last frame)
            placed.append(np.linalg.norm((ep["c2"][-1] - ep["c1"][-1])[:2]) < 0.02)
    placed = np.array(placed)
    xy_r = np.array(xy_r)
    print(f"  {name}: episodes {len(run)}")
    print(f"    at GRASP   gripper-to-cube_2 distance:", q(reach_g))
    print(f"    at RELEASE gripper-to-cube_1 xy offset:", q(xy_r), " z:", q(z_r))
    print(f"    cube_2 placed on cube_1 at end: {placed.sum()}/{len(placed)} = {placed.mean()*100:.1f}%")
    print(f"      release xy offset | placed:", q(xy_r[placed]), "| not placed:", q(xy_r[~placed]))

# paired: same scene, difference vs reference at release, split by placement outcome of the arc run
print()
print("  PAIRED (same scene) at RELEASE: |arc gripper - reference gripper|, split by arc placement outcome")
dpos, placed = [], []
for k, a in arc.items():
    rr = ref.get(k)
    if rr is None:
        continue
    ga, ra = events(a["grip"]); gr, rf = events(rr["grip"])
    if ra is None or rf is None:
        continue
    dpos.append(np.linalg.norm(a["pos"][ra] - rr["pos"][rf]))
    placed.append(np.linalg.norm((a["c2"][-1] - a["c1"][-1])[:2]) < 0.02)
dpos, placed = np.array(dpos), np.array(placed)
print("    arc placed:    ", q(dpos[placed]))
print("    arc not placed:", q(dpos[~placed]))
