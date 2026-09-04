"""Does how much of the arc the arm traces depend on the arc's DIRECTION relative to the arm?

If the attenuation were purely dynamic (a low-pass on a fast bump), it would be the same in every
direction and would shrink when the bump is slowed. If it is kinematic -- the arm near a singular
posture, the damped-least-squares IK attenuating motion along the arm -- it depends on the angle
between the arc direction and the base-to-eef radial axis, and does not shrink with time.

Per subtask segment: arc direction = (perturbed - unperturbed) command at the envelope peak;
radial axis = eef position at that frame (the robot base is at the env origin); ratio = achieved
peak / commanded peak from effective_amplitude's excess-distance measure. Binned by |cos(angle)|
and by the vertical component |d_z|.

usage: arc_direction_dependence.py <out_dir> <tag> [<tag> ...]
"""
import glob
import os
import sys

import h5py
import numpy as np

CUBES = ("cube_1", "cube_2", "cube_3")
OPEN_VAL = 0.04


def transitions(grip):
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = jaw < OPEN_VAL - 0.001
    return np.where(np.diff(closed.astype(int)) != 0)[0] + 1


def dist_to_polyline(pts, poly):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    ab2 = (ab * ab).sum(axis=1) + 1e-12
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(axis=2) / ab2[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2).min(axis=1)


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
        rows = []
        for suffix in ("", "_failed"):
            p = os.path.join(out, f"ch_{tag}{suffix}.hdf5")
            if not os.path.exists(p):
                continue
            with h5py.File(p, "r") as f:
                for k in f["data"]:
                    d = f["data"][k]
                    ro = d["states"]["rigid_object"]
                    key = np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]).astype(float)
                    if len(keys) == 0:
                        break
                    dist = np.linalg.norm(keys - key, axis=1)
                    i = int(np.argmin(dist))
                    if dist[i] > 2e-3:
                        continue
                    pos = d["obs/eef_pos"][:]
                    ev = transitions(d["obs/gripper_pos"][:])
                    segs = sorted(recs[i], key=lambda r: r[0])[:4]
                    nseg = min(len(segs), len(ev))
                    bounds = [0] + list(ev[:nseg])
                    for s in range(nseg):
                        _, unp, per = segs[s]
                        seg = pos[bounds[s]:bounds[s + 1]]
                        if len(seg) < 12 or len(unp) < 2 or len(per) != len(unp) and len(per) < len(unp):
                            pass
                        # arc direction at the peak of the commanded offset
                        n = min(len(unp), len(per))
                        off = per[:n] - unp[:n] if len(per) == len(unp) else None
                        if off is None:
                            # stretched runs: perturbed has more frames; align by nearest polyline point instead
                            dpp = dist_to_polyline(per, unp)
                            j = int(np.argmax(dpp))
                            cmd_peak = dpp[j] * 100
                            # direction: from the nearest unperturbed point to the perturbed point
                            a, b = unp[:-1], unp[1:]
                            ab = b - a
                            t = np.clip(((per[j] - a) * ab).sum(1) / ((ab * ab).sum(1) + 1e-12), 0, 1)
                            proj = a + t[:, None] * ab
                            m = int(np.argmin(np.linalg.norm(per[j] - proj, axis=1)))
                            dvec = per[j] - proj[m]
                            peak_pos = per[j]
                        else:
                            mag = np.linalg.norm(off, axis=1)
                            j = int(np.argmax(mag))
                            cmd_peak = mag[j] * 100
                            dvec = off[j]
                            peak_pos = per[j]
                        if cmd_peak < 0.3 or len(seg) < 12:
                            continue
                        dvec = dvec / (np.linalg.norm(dvec) + 1e-9)
                        radial = peak_pos / (np.linalg.norm(peak_pos) + 1e-9)
                        d_unp = dist_to_polyline(seg[6:], unp) * 100
                        d_per = dist_to_polyline(seg[6:], per) * 100
                        ach = (d_unp - d_per).max()
                        rows.append((abs(float(dvec @ radial)), abs(float(dvec[2])), ach / cmd_peak, s))
        if not rows:
            print(f"=== {tag}: no matched segments ===")
            continue
        r = np.array(rows)
        print(f"=== {tag}: {len(r)} segments, overall achieved/commanded median {np.median(r[:, 2]):.2f} ===")
        print("  by |cos(arc, radial base->eef)|:")
        for lo, hi in ((0.0, 0.33), (0.33, 0.66), (0.66, 1.01)):
            m = (r[:, 0] >= lo) & (r[:, 0] < hi)
            if m.sum():
                print(f"     {lo:.2f}-{hi:.2f}: ratio median {np.median(r[m, 2]):.2f}  (n={m.sum()})")
        print("  by |vertical component d_z|:")
        for lo, hi in ((0.0, 0.33), (0.33, 0.66), (0.66, 1.01)):
            m = (r[:, 1] >= lo) & (r[:, 1] < hi)
            if m.sum():
                print(f"     {lo:.2f}-{hi:.2f}: ratio median {np.median(r[m, 2]):.2f}  (n={m.sum()})")
        print("  by subtask:", {int(s): round(float(np.median(r[r[:, 3] == s, 2])), 2) for s in np.unique(r[:, 3])})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
