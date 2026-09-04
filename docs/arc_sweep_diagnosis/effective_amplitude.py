"""How big is the arc the arm ACTUALLY traces, against the command it was given?

The arm executes ~15-20% of each commanded step, so a 10 cm commanded bump comes out smaller.
"1.2 cm" was quoted as the perturbation for two versions before anyone integrated the envelope; this
script reports the achieved deviation instead, measured against the run's own commanded paths.

Inputs: the run's dataset (ch_<tag>.hdf5 / ch_<tag>_failed.hdf5) and the commanded-path dump the
trial script writes when CMD_DIR is set (<out>/cmd_<tag>/*.npz: per subtask, the object-centric
transform's output = unperturbed command, and the sequence handed to from_poses = perturbed
command, keyed by scene layout).

Per subtask segment (achieved path between gripper transitions, first 6 frames skipped), the
achieved arc at each point is (distance to the UNPERTURBED commanded polyline) minus (distance to
the PERTURBED one). Distances are spatial, so lag along the path does not count; and taking the
difference cancels the tracking lag that dominates raw distances (the first subtask's command
starts 17 cm from the arm and is bridged in 5 frames). On a 0 cm run the two commands coincide and
the measure is identically zero. Reported per run: commanded peak (perturbed vs unperturbed
command), achieved peak and mean, the ratio, and the raw distance to the unperturbed command left
at the gripper transition.

usage: effective_amplitude.py <out_dir> <tag> [<tag> ...]
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


def key_of(v):
    return tuple(np.round(np.asarray(v, float), 3))


def dist_to_polyline(pts, poly):
    """pts (N,3), poly (M,3) -> (N,) distance to the closest polyline segment."""
    a, b = poly[:-1], poly[1:]                         # (M-1, 3)
    ab = b - a
    ab2 = (ab * ab).sum(axis=1) + 1e-12
    ap = pts[:, None, :] - a[None, :, :]               # (N, M-1, 3)
    t = np.clip((ap * ab[None]).sum(axis=2) / ab2[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2).min(axis=1)


def load_cmds(out, tag):
    """Returns (keys array (K,9), list of per-key record lists)."""
    keys, recs = [], []
    index = {}
    for f in sorted(glob.glob(os.path.join(out, f"cmd_{tag}", "*.npz"))):
        z = np.load(f)
        k = key_of(z["key"])
        if k not in index:
            index[k] = len(keys)
            keys.append(np.asarray(z["key"], float))
            recs.append([])
        recs[index[k]].append((int(z["subtask"]), z["unperturbed"], z["perturbed"]))
    return np.stack(keys) if keys else np.zeros((0, 9)), recs


def load_eps(out, tag):
    eps = []
    for suffix, ok in (("", True), ("_failed", False)):
        p = os.path.join(out, f"ch_{tag}{suffix}.hdf5")
        if not os.path.exists(p):
            continue
        with h5py.File(p, "r") as f:
            for k in f["data"]:
                d = f["data"][k]
                ro = d["states"]["rigid_object"]
                eps.append({
                    "key": np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]).astype(float),
                    "ok": ok, "p": d["obs/eef_pos"][:], "ev": transitions(d["obs/gripper_pos"][:]),
                })
    return eps


def main(out, tags):
    print(f"  {'run':<10s} {'episodes':>9s} {'cmd peak':>9s} {'achieved peak':>16s} {'achieved mean':>14s} "
          f"{'ach/cmd':>8s} {'at contact':>11s}")
    for tag in tags:
        keys, all_recs = load_cmds(out, tag)
        eps = load_eps(out, tag)
        cmd_peak, ach_peak, ach_mean, at_contact = [], [], [], []
        matched = 0
        for e in eps:
            if len(keys) == 0:
                break
            dist = np.linalg.norm(keys - e["key"], axis=1)
            i = int(np.argmin(dist))
            if dist[i] > 2e-3:
                continue
            recs = sorted(all_recs[i], key=lambda r: r[0])[:4]
            nseg = min(len(recs), len(e["ev"]))
            if nseg == 0:
                continue
            matched += 1
            bounds = [0] + list(e["ev"][:nseg])
            for s in range(nseg):
                _, unp, per = recs[s]
                seg = e["p"][bounds[s]:bounds[s + 1]]
                if len(seg) < 12 or len(unp) < 2:
                    continue
                cmd_peak.append(dist_to_polyline(per, unp).max() * 100)
                # The arm trails its command by up to ~18 cm right after the interpolation bridge
                # into the segment, so the raw distance to the unperturbed command is dominated by
                # lag, not by the arc (11.8 cm median on a 0 cm run). The arc is what the distance
                # to the unperturbed command exceeds the distance to the perturbed command by:
                # identically zero when the two commands coincide, and equal to the bump where the
                # arm follows it.
                d_unp = dist_to_polyline(seg[6:], unp) * 100
                d_per = dist_to_polyline(seg[6:], per) * 100
                excess = d_unp - d_per
                ach_peak.append(excess.max())
                ach_mean.append(excess.mean())
                at_contact.append(d_unp[-1])
        if not ach_peak:
            print(f"  {tag:<10s} {matched:9d}   (no matched segments)")
            continue
        cp, ap, am, ac = map(np.array, (cmd_peak, ach_peak, ach_mean, at_contact))
        ratio = np.median(ap) / np.median(cp) if np.median(cp) > 0.05 else float("nan")
        print(f"  {tag:<10s} {matched:9d} {np.median(cp):7.2f}cm {np.median(ap):7.2f} / p90 {np.percentile(ap, 90):5.2f}"
              f" {np.median(am):7.2f} / {np.percentile(am, 90):5.2f} {ratio:8.2f} {np.median(ac):9.2f}cm")
    print("  (cm; median / p90 over matched subtask segments; 'at contact' = deviation at the gripper transition)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
