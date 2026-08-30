"""Does the residual accumulate from one subtask to the next?

At the first grasp the gripper is 0.72 cm away from where the unperturbed run had it; by the end of
the episode it is 1.21 cm. Each subtask draws its own independent arc direction, and nothing resets
the arm's configuration between them, so the drifts should add like a random walk -- four subtasks
would give roughly a doubling, which is about what those two numbers show.

The four contact events are found from the gripper itself rather than assumed: the jaws close on
cube_2, open to release it, close on cube_3, open again. Measuring the residual at each says whether
it grows monotonically, and therefore whether a longer task would suffer more.
"""

import sys

import h5py
import numpy as np

ROOT = "/home/pk/IsaacLab/datasets/perturbation_sweep_v2_arc"
CUBES = ("cube_1", "cube_2", "cube_3")
OPEN_VAL = 0.04
CLOSED_MARGIN = 0.001


def key_of(ro):
    return tuple(np.round(np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]), 6))


def contact_events(grip):
    """Frames where the jaws change state: close, open, close, open."""
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = jaw < OPEN_VAL - CLOSED_MARGIN
    flips = np.where(np.diff(closed.astype(int)) != 0)[0] + 1
    return flips


def load_run(seed, cm):
    tag = f"cm{cm:.1f}".replace(".", "p")
    out = {}
    for suffix, ok in (("", True), ("_failed", False)):
        with h5py.File(f"{ROOT}/seed{seed}/generated_{tag}{suffix}.hdf5", "r") as f:
            d = f["data"]
            for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
                obs = d[k]["obs"]
                out[key_of(d[k]["states"]["rigid_object"])] = {
                    "success": ok,
                    "pos": obs["eef_pos"][:],
                    "grip": obs["gripper_pos"][:],
                }
    return out


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ref = load_run(seed, 0.5)
    print(f"seed={seed}, paired against 0.5 cm, scenes succeeding at both\n")
    print(f"  {'cm':>5s} {'n':>5s} " + " ".join(f"{f'event {i + 1}':>10s}" for i in range(4)))
    for cm in (1.0, 2.0, 3.0):
        other = load_run(seed, cm)
        per_event = [[] for _ in range(4)]
        n = 0
        for k, a in other.items():
            r = ref.get(k)
            if r is None or not (a["success"] and r["success"]):
                continue
            ea, er = contact_events(a["grip"]), contact_events(r["grip"])
            if len(ea) < 4 or len(er) < 4:
                continue
            n += 1
            for i in range(4):
                per_event[i].append(float(np.linalg.norm(a["pos"][ea[i]] - r["pos"][er[i]])))
        print(f"  {cm:5.1f} {n:5d} " + " ".join(f"{np.median(v) * 100:9.3f}cm" for v in per_event))

    print("\n  events: 1 close on cube_2, 2 release it, 3 close on cube_3, 4 release it.")
    print("  a monotone rise means each subtask adds drift the next one inherits.")


if __name__ == "__main__":
    main()
