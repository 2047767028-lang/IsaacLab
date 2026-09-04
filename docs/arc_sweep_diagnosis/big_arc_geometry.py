"""Before simulating a 10 cm arc: how often would an isotropic arc of that size put the end-effector
below the table top, or newly within one cube edge of a cube it is not interacting with?

Offline, on the 10 source demos' achieved eef paths, subtask by subtask (segments between gripper
transitions), v3 envelope (order 6, centred peak, freeze_frac 0.3), 2000 random directions per
segment. "Newly near" counts only frames where the unperturbed path is farther than one cube edge
from that cube; the cube being carried is excluded.
"""
import h5py
import numpy as np

SRC = "/home/pk/IsaacLab/datasets/annotated_dataset.hdf5"
OPEN_VAL = 0.04
CUBE = 0.0468
rng = np.random.default_rng(0)


def transitions(grip):
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = jaw < OPEN_VAL - 0.001
    return np.where(np.diff(closed.astype(int)) != 0)[0] + 1


def envelope(n, freeze=0.3, order=6.0, peak=0.5):
    free = int(round(n * (1 - freeze)))
    u = np.clip(np.linspace(0, 1, free), 1e-6, 1 - 1e-6)
    a, b = order * peak, order * (1 - peak)
    raw = u**a * (1 - u) ** b
    env = np.zeros(n)
    env[:free] = raw / (peak**a * (1 - peak) ** b)
    return env


def sample_dirs(k, mode):
    if mode == "cone":           # elevation +45..+70 deg, the band the human demos occupy
        el = np.radians(rng.uniform(45, 70, size=k))
        az = rng.uniform(0, 2 * np.pi, size=k)
        return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    d = rng.normal(size=(k, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    if mode == "upper":
        d[:, 2] = np.abs(d[:, 2])
    return d


# subtask k: target object, carried object
TARGET = ["cube_2", "cube_1", "cube_3", "cube_2"]
CARRIED = [None, "cube_2", None, "cube_3"]

segs = []
with h5py.File(SRC, "r") as f:
    for k in f["data"]:
        d = f["data"][k]
        pos = d["obs/eef_pos"][:]
        ev = transitions(d["obs/gripper_pos"][:])
        cubes = {c: d[f"states/rigid_object/{c}/root_pose"][:, :3] for c in ("cube_1", "cube_2", "cube_3")}
        table_top = cubes["cube_1"][0, 2] - CUBE / 2
        bounds = [0] + [int(e) for e in ev] + [len(pos)]
        for i in range(min(4, len(bounds) - 1)):
            s, e = bounds[i], bounds[i + 1]
            if e - s < 10:
                continue
            others = [c for c in cubes if c != TARGET[i] and c != CARRIED[i]]
            segs.append({"path": pos[s:e], "table": table_top, "k": i,
                         "others": [cubes[c][s:e] for c in others]})

print(f"{len(segs)} segments. eef height above table top (cm): "
      f"start median {np.median([sg['path'][0, 2] - sg['table'] for sg in segs]) * 100:.1f}, "
      f"at envelope peak median {np.median([sg['path'][int(round(0.35 * len(sg['path']))), 2] - sg['table'] for sg in segs]) * 100:.1f}, "
      f"path min median {np.median([sg['path'][:, 2].min() - sg['table'] for sg in segs]) * 100:.1f}")
by_k = {}
for sg in segs:
    by_k.setdefault(sg["k"], []).append(sg["path"][int(round(0.35 * len(sg["path"]))), 2] - sg["table"])
print("  height at the peak frame by subtask (median cm):", {k: round(float(np.median(v)) * 100, 1) for k, v in sorted(by_k.items())})
print()
print(f"{'amp':>5s} {'dirs':>6s} {'below table':>12s} {'newly near other cube':>22s} {'either':>8s}")
for amp in (0.03, 0.05, 0.08, 0.10):
    for mode in ("iso", "upper", "cone"):
        below = near = either = total = 0
        for sg in segs:
            n = len(sg["path"])
            env = envelope(n)
            dirs = sample_dirs(2000, mode)
            P = sg["path"][None] + amp * env[None, :, None] * dirs[:, None, :]      # (k, n, 3)
            b = (P[:, :, 2] < sg["table"] + 0.005).any(axis=1)
            nr = np.zeros(len(dirs), bool)
            for oc in sg["others"]:
                base_far = np.linalg.norm(sg["path"] - oc, axis=1) > CUBE            # (n,)
                dist = np.linalg.norm(P - oc[None], axis=2)                          # (k, n)
                nr |= ((dist < CUBE) & base_far[None, :]).any(axis=1)
            below += b.sum(); near += nr.sum(); either += (b | nr).sum(); total += len(dirs)
        print(f"{amp*100:4.0f}cm {mode:>6s} {100*below/total:11.1f}% {100*near/total:21.1f}% {100*either/total:7.1f}%")
