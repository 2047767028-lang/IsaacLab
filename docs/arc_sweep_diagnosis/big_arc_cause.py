"""Why does the arm trace only ~42% of a commanded arc, and why does the gate stop converging at
large amplitude? Four checks on the recorded runs (needs the CMD_DIR dumps for the first).

1. Per-frame tracking gain on subtask 0, where achieved frame t aligns with command frame t-s for a
   small shift s (interpolation bridge) found per episode: the achieved step projected on the
   error direction, divided by the error. Binned by error size: a constant ratio means a linear
   low-pass; a falling ratio means velocity/effort saturation.
2. Joint-limit proximity: fraction of frames (free zone, and at the gripper transitions) with any
   arm joint within 0.05 rad of its Franka limit.
3. Joint-velocity saturation: fraction of frames with |q_dot| above 90% of the Franka velocity
   limit (2.175 rad/s joints 1-4, 2.61 joints 5-7).
4. Cube seat: eef-to-cube_2 xy offset at the first release, and the cube's lateral speed during
   the subtask-1 carry (a cube slipping in the jaws under lateral acceleration).

usage: big_arc_cause.py <out_dir> <tag> [<tag> ...]
"""
import glob
import os
import sys

import h5py
import numpy as np

CUBES = ("cube_1", "cube_2", "cube_3")
OPEN_VAL = 0.04
Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
V_LIM = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])


def transitions(grip):
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = jaw < OPEN_VAL - 0.001
    return np.where(np.diff(closed.astype(int)) != 0)[0] + 1


def load_cmds(out, tag):
    keys, recs, index = [], [], {}
    for f in sorted(glob.glob(os.path.join(out, f"cmd_{tag}", "*.npz"))):
        z = np.load(f)
        k = tuple(np.round(np.asarray(z["key"], float), 3))
        if k not in index:
            index[k] = len(keys); keys.append(np.asarray(z["key"], float)); recs.append([])
        recs[index[k]].append((int(z["subtask"]), z["unperturbed"], z["perturbed"]))
    return (np.stack(keys) if keys else np.zeros((0, 9))), recs


def main(out, tags):
    for tag in tags:
        keys, recs = load_cmds(out, tag)
        gains = {b: [] for b in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 30))}
        near_free, near_contact, vsat_free, n_free, n_contact = 0, 0, 0, 0, 0
        seat, carry_v = [], []
        n_eps = 0
        for suffix in ("", "_failed"):
            p = os.path.join(out, f"ch_{tag}{suffix}.hdf5")
            if not os.path.exists(p):
                continue
            with h5py.File(p, "r") as f:
                for k in f["data"]:
                    d = f["data"][k]
                    ro = d["states"]["rigid_object"]
                    pos, q, qd = d["obs/eef_pos"][:], d["obs/joint_pos"][:, :7], d["obs/joint_vel"][:, :7]
                    ev = transitions(d["obs/gripper_pos"][:])
                    c2 = ro["cube_2"]["root_pose"][:, :3]
                    v2 = ro["cube_2"]["root_velocity"][:, :3]
                    n_eps += 1
                    # 2/3: limits and velocity, free zone = first 60% of each segment, contact = transition frames
                    bounds = [0] + list(ev[:4])
                    for s in range(len(bounds) - 1):
                        a, b = bounds[s], bounds[s + 1]
                        fz = slice(a, a + int(0.6 * (b - a)))
                        near = ((q[fz] - Q_LO) < 0.05) | ((Q_HI - q[fz]) < 0.05)
                        near_free += near.any(axis=1).sum(); n_free += near.shape[0]
                        vsat_free += (np.abs(qd[fz]) > 0.9 * V_LIM).any(axis=1).sum()
                    for t in ev[:4]:
                        n_contact += 1
                        near_contact += int((((q[t] - Q_LO) < 0.05) | ((Q_HI - q[t]) < 0.05)).any())
                    # 4: seat at first release and cube lateral speed while carried
                    if len(ev) >= 2:
                        seat.append(np.linalg.norm((pos[ev[1]] - c2[ev[1]])[:2]) * 100)
                        carry_v.append(np.linalg.norm(v2[ev[0] + 5:ev[1] - 5, :2], axis=1).max() * 100 if ev[1] - ev[0] > 12 else np.nan)
                    # 1: tracking gain on subtask 0 against the perturbed command
                    if len(keys) == 0 or len(ev) == 0:
                        continue
                    key = np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]).astype(float)
                    dist = np.linalg.norm(keys - key, axis=1)
                    i = int(np.argmin(dist))
                    if dist[i] > 2e-3:
                        continue
                    r0 = [r for r in recs[i] if r[0] == 0]
                    if not r0:
                        continue
                    cmd = r0[0][2]                              # perturbed command, subtask 0
                    seg = pos[:ev[0]]
                    best, best_s = None, 0
                    for s in range(0, 12):
                        m = min(len(cmd), len(seg) - s - 1)
                        if m < 15:
                            continue
                        e = np.linalg.norm(cmd[:m] - seg[s:s + m], axis=1).mean()
                        if best is None or e < best:
                            best, best_s = e, s
                    if best is None:
                        continue
                    s = best_s
                    m = min(len(cmd), len(seg) - s - 1)
                    err = cmd[:m] - seg[s:s + m]                # error vector at each aligned frame
                    step = seg[s + 1:s + m + 1] - seg[s:s + m]  # achieved displacement next frame
                    en = np.linalg.norm(err, axis=1) + 1e-9
                    proj = (step * err).sum(axis=1) / en        # step along the error direction
                    for j in range(m):
                        e_cm = en[j] * 100
                        for (lo, hi) in gains:
                            if lo <= e_cm < hi:
                                gains[(lo, hi)].append(proj[j] / en[j])
        print(f"=== {tag}: {n_eps} episodes ===")
        print("  1. tracking gain (achieved step along error / error), subtask 0, by error size:")
        for (lo, hi), g in gains.items():
            if g:
                g = np.array(g)
                print(f"       error {lo:2d}-{hi:2d} cm: gain median {np.median(g):.3f}  (n={len(g)}); "
                      f"achieved step median {np.median(g) * (lo + hi) / 2:.2f} cm/frame at bin centre")
        print(f"  2. joint within 0.05 rad of a limit: free zone {100 * near_free / max(n_free, 1):.1f}% of frames, "
              f"at contact {100 * near_contact / max(n_contact, 1):.1f}% of events")
        print(f"  3. joint speed above 90% of limit in the free zone: {100 * vsat_free / max(n_free, 1):.2f}% of frames")
        seat, carry_v = np.array(seat), np.array(carry_v)
        print(f"  4. cube seat at first release: median {np.nanmedian(seat):.2f} / p90 {np.nanpercentile(seat, 90):.2f} cm;"
              f" cube lateral speed while carried: median {np.nanmedian(carry_v):.1f} / p90 {np.nanpercentile(carry_v, 90):.1f} cm/s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
