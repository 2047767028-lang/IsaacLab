"""How far is the SOURCE demo's arm from its own reconstructed target, frame by frame?

`action_to_target_eef_pose` reconstructs target_pos = curr_pos + action[:3] (unscaled), so
|target - achieved| in the source demo is exactly |action[:3]| in metres. Compare that with the
achieved per-step displacement to get the source's own "execution fraction", and look at both at
the grasp frame specifically (first frame the jaws leave the open value by >1mm).
"""
import h5py
import numpy as np

F = "/home/pk/IsaacLab/datasets/annotated_dataset.hdf5"
OPEN_VAL = 0.04

with h5py.File(F, "r") as f:
    d0 = f["data/demo_0"]
    print("keys under demo_0:")
    def walk(g, p=""):
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                print(f"  {p}{k}: {v.shape} {v.dtype}")
            else:
                walk(v, p + k + "/")
    walk(d0)
    print("env_args:", str(f["data"].attrs.get("env_args", ""))[:400])
    print()

    all_lag, all_step, grasp_lag, grasp_step, all_ratio = [], [], [], [], []
    for k in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])):
        d = f["data"][k]
        act = d["actions"][:]
        pos = d["obs/eef_pos"][:]
        grip = d["obs/gripper_pos"][:]
        lag = np.linalg.norm(act[:, :3], axis=1)            # |target - achieved| by construction
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)  # achieved displacement per step
        all_lag.append(lag[:-1]); all_step.append(step)
        jaw = np.minimum(grip[:, 0], -grip[:, 1])
        closed = np.where(jaw < OPEN_VAL - 0.001)[0]
        g = int(closed[0]) if len(closed) else None
        if g is not None and g < len(step):
            grasp_lag.append(lag[g]); grasp_step.append(step[g])
        print(f"{k}: T={len(act)}  |act_pos| median={np.median(lag)*100:.2f}cm p90={np.percentile(lag,90)*100:.2f}cm"
              f"  achieved step median={np.median(step)*100:.2f}cm  grasp frame={g}"
              f"  lag@grasp={lag[g]*100:.2f}cm  step@grasp={step[g]*100:.3f}cm" if g is not None else f"{k}: no grasp")
    lag = np.concatenate(all_lag); step = np.concatenate(all_step)
    print()
    print(f"ALL FRAMES  |target-achieved| median {np.median(lag)*100:.2f}cm  p90 {np.percentile(lag,90)*100:.2f}cm")
    print(f"            achieved step     median {np.median(step)*100:.2f}cm  p90 {np.percentile(step,90)*100:.2f}cm")
    print(f"            execution fraction (median step / median lag) = {np.median(step)/np.median(lag):.3f}")
    print(f"GRASP FRAME |target-achieved| median {np.median(grasp_lag)*100:.2f}cm  (n={len(grasp_lag)})")
    print(f"            achieved step     median {np.median(grasp_step)*100:.3f}cm")
