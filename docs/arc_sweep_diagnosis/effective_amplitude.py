"""How big is the arc the arm ACTUALLY traces, against the nominal PERTURB_ARC_STD?

The arm executes ~15-20% of each commanded step, so a 10 cm commanded bump comes out smaller.
"1.2 cm" was quoted as the perturbation for two versions before anyone integrated the envelope; this
script reports the achieved deviation instead.

The noise floor is read off by including a run with (almost) no arc among the arc tags: its
deviation from the reference is what two runs with different noise realisations differ by anyway.

Method: pair each arc episode with the 0 cm reference episode on the same scene (reseeded runs,
layout match < 2 mm). Cut both into segments at the gripper transitions, resample each segment's
eef path to 100 points of normalised time, and take |p_arc(u) - p_ref(u)|. The commanded free zone is
the first 70% of the commanded segment; the recorded segment also contains the hold before the
transition, so the free zone maps to roughly u < 0.6. Reported: the maximum deviation over u < 0.6
(achieved peak), the mean over u < 0.6 (path-integrated), and the deviation at u = 0.95 (what is
left when the gripper acts).

usage: effective_amplitude.py <out_dir> <ref_tag> <arc_tag> [<arc_tag> ...]
"""
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


def load(out, tag):
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
                    "key": np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]),
                    "ok": ok,
                    "p": d["obs/eef_pos"][:],
                    "ev": transitions(d["obs/gripper_pos"][:]),
                })
    return eps


def resample(seg, n=100):
    t = np.linspace(0, 1, len(seg))
    u = np.linspace(0, 1, n)
    return np.stack([np.interp(u, t, seg[:, i]) for i in range(3)], axis=1)


def main(out, ref_tag, arc_tags):
    ref = load(out, ref_tag)
    keys = np.stack([e["key"] for e in ref])
    nominal = {"gt_zero": 0.0, "gt_ref": 0.5, "gt_arc": 3.0, "big5": 5.0, "big8": 8.0, "big10": 10.0,
               "op_arc_hold": 1.2, "arc_hold": 3.0}
    print(f"reference {ref_tag}: {len(ref)} episodes")
    print(f"  {'run':<10s} {'nominal':>8s} {'pairs':>6s} {'achieved peak (u<0.6)':>22s} {'mean over u<0.6':>16s} "
          f"{'at u=0.95':>10s} {'peak/nominal':>13s}")
    for tag in arc_tags:
        arc = load(out, tag)
        peak, mean, late = [], [], []
        pairs = 0
        for a in arc:
            dist = np.linalg.norm(keys - a["key"], axis=1)
            i = int(np.argmin(dist))
            if dist[i] > 2e-3:
                continue
            r = ref[i]
            nseg = min(len(a["ev"]), len(r["ev"]), 4)
            if nseg < 1:
                continue
            pairs += 1
            ba = [0] + list(a["ev"][:nseg])
            br = [0] + list(r["ev"][:nseg])
            for s in range(nseg):
                sa, sr = a["p"][ba[s]:ba[s + 1]], r["p"][br[s]:br[s + 1]]
                if len(sa) < 5 or len(sr) < 5:
                    continue
                diff = resample(sa) - resample(sr)                      # (100, 3)
                # Action noise is zero-mean and decorrelates within a few frames; the arc is a
                # single-direction bump spanning ~40 frames. A 9-sample moving average on the
                # deviation vector suppresses the former and keeps the latter.
                k = np.ones(9) / 9.0
                diff = np.stack([np.convolve(diff[:, i], k, mode="same") for i in range(3)], axis=1)
                dlt = np.linalg.norm(diff, axis=1) * 100
                peak.append(dlt[:60].max())
                mean.append(dlt[:60].mean())
                late.append(dlt[95])
        peak, mean, late = map(np.array, (peak, mean, late))
        nom = nominal.get(tag, float("nan"))
        ratio = np.median(peak) / nom if nom > 0 else float("nan")
        print(f"  {tag:<10s} {nom:7.1f}cm {pairs:6d} {np.median(peak):9.2f} / p90 {np.percentile(peak, 90):5.2f} cm"
              f" {np.median(mean):8.2f} / {np.percentile(mean, 90):5.2f} {np.median(late):9.2f} {ratio:12.2f}")
    print("  (median / p90 over all paired segments; 'at u=0.95' is just before the gripper acts)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
