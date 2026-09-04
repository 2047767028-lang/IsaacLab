"""How big is the arc the arm ACTUALLY traces, against the command it was given?

The arm executes ~15-20% of each commanded step, so a 10 cm commanded bump comes out smaller.
"1.2 cm" was quoted as the perturbation for two versions before anyone integrated the envelope; this
script reports the achieved deviation instead, measured against the run's own commanded paths.

Inputs: the run's dataset (ch_<tag>.hdf5 / ch_<tag>_failed.hdf5) and the commanded-path dump the
trial script writes when CMD_DIR is set (<out>/cmd_<tag>/*.npz: per subtask, the object-centric
transform's output = unperturbed command, and the sequence handed to from_poses = perturbed
command, keyed by scene layout).

Per subtask segment (achieved path between gripper transitions, first 6 frames skipped to leave out
the interpolation bridge from the previous subtask), the deviation of every achieved point is its
distance to the UNPERTURBED commanded polyline -- a spatial measure, so the arm's lag along the
path does not count. Reported per run: commanded peak (perturbed vs unperturbed command),
achieved peak and mean, the ratio, and the deviation left at the gripper transition. A 0 cm run
gives the floor: what noise and tracking alone produce.

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
    cmds = {}
    for f in sorted(glob.glob(os.path.join(out, f"cmd_{tag}", "*.npz"))):
        z = np.load(f)
        cmds.setdefault(key_of(z["key"]), []).append((int(z["subtask"]), z["unperturbed"], z["perturbed"]))
    return cmds


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
                    "key": key_of(np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES])),
                    "ok": ok, "p": d["obs/eef_pos"][:], "ev": transitions(d["obs/gripper_pos"][:]),
                })
    return eps


def main(out, tags):
    print(f"  {'run':<10s} {'episodes':>9s} {'cmd peak':>9s} {'achieved peak':>16s} {'achieved mean':>14s} "
          f"{'ach/cmd':>8s} {'at contact':>11s}")
    for tag in tags:
        cmds = load_cmds(out, tag)
        eps = load_eps(out, tag)
        cmd_peak, ach_peak, ach_mean, at_contact = [], [], [], []
        matched = 0
        for e in eps:
            recs = cmds.get(e["key"])
            if not recs:
                continue
            recs = sorted(recs, key=lambda r: r[0])[:4]
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
                dev = dist_to_polyline(seg[6:], unp) * 100
                ach_peak.append(dev.max())
                ach_mean.append(dev.mean())
                at_contact.append(dev[-1])
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
