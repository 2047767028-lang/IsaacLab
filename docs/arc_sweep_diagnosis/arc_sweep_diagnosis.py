"""Why does generation success fall with arc amplitude, when the contact segment is untouched?

The hypothesis under test: the sin^2-family envelope is zero in value *and* slope at both ends of
the free zone, and the trailing freeze_frac of every subtask is byte-identical to the source, so the
contact phase should be unaffected and the success criterion must be treating the arc group
unfairly.

That premise is about the TARGET pose sequence, which is verifiably unchanged. It says nothing about
the ACHIEVED state: the arm tracks those targets through a differential IK controller, and a 7-DOF
arm can reach the same end-effector pose in different null-space configurations. Three measurements
separate the readings.

M1 -- placement accuracy among SUCCESSES only. If the contact phase is genuinely unaffected, the
     distance from each placed cube to the one below it should not depend on amplitude. Successes
     only, so it cannot be confounded by which episodes failed.

M2 -- where the lost successes go. Replay the criterion's parts over the FAILURES and split them by
     which stage broke, at low amplitude versus high.

M3 -- does the criterion's known in-flight loophole favour the arc group or penalise it? The check
     fires on an instantaneous configuration, so a cube passing through the stacked pose while
     falling counts as a success. If that loophole is doing more work at high amplitude, the stock
     criterion is being *generous* to arc, not unfair.

Usage:  python arc_sweep_diagnosis.py [seed] [cm ...]
"""

import sys

import h5py
import numpy as np

ROOT = "/home/pk/IsaacLab/datasets/perturbation_sweep_v2_arc"
XY_TH, H_TH, H_DIFF = 0.04, 0.005, 0.0468
OPEN_VAL = 0.04
GRIP_TOL = 1e-4 + 1e-4 * OPEN_VAL
CUBES = ("cube_1", "cube_2", "cube_3")


def tag(cm: float) -> str:
    return f"cm{cm:.1f}".replace(".", "p")


def path(seed: int, cm: float, failed: bool) -> str:
    return f"{ROOT}/seed{seed}/generated_{tag(cm)}{'_failed' if failed else ''}.hdf5"


def geom_parts(pos):
    """Per-pair booleans plus the raw distances, from (T,3) cube positions."""
    c1, c2, c3 = pos
    out = {}
    for name, (a, b) in (("12", (c1, c2)), ("23", (c2, c3))):
        dd = a - b
        xy = np.linalg.norm(dd[:, :2], axis=1)
        h = np.abs(dd[:, 2])
        out[name] = {
            "xy": xy,
            "dz": dd[:, 2],
            "ok": (xy < XY_TH) & (h - H_DIFF < H_TH) & (dd[:, 2] < 0.0),
        }
    out["both"] = out["12"]["ok"] & out["23"]["ok"]
    return out


def load(p, limit=None):
    eps = []
    with h5py.File(p, "r") as f:
        d = f["data"]
        keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
        if limit:
            keys = keys[:limit]
        for k in keys:
            st = d[k]["states"]
            pos = [st["rigid_object"][c]["root_pose"][:, :3] for c in CUBES]
            spd = np.stack(
                [np.linalg.norm(st["rigid_object"][c]["root_velocity"][:, :3], axis=1) for c in CUBES]
            ).max(axis=0)
            jp = st["articulation"]["robot"]["joint_position"][:]
            jaw = np.maximum(np.abs(jp[:, 7] - OPEN_VAL), np.abs(jp[:, 8] - OPEN_VAL))
            eps.append({"k": k, "g": geom_parts(pos), "spd": spd, "jaw": jaw, "pos": pos})
    return eps


def m1_placement(seed, cms):
    print("\n=== M1: placement accuracy among SUCCESSES (does the contact phase degrade?) ===")
    print(f"  {'cm':>5s} {'n':>4s} {'xy(c1,c2) med':>14s} {'p90':>7s} {'xy(c2,c3) med':>14s} {'p90':>7s}")
    for cm in cms:
        eps = load(path(seed, cm, failed=False))
        # the frame the criterion actually accepted on -- last qualifying frame, i.e. the settled one
        xy12, xy23 = [], []
        for e in eps:
            q = e["g"]["both"] & (e["jaw"] <= GRIP_TOL)
            if not q.any():
                continue
            i = np.where(q)[0][-1]
            xy12.append(e["g"]["12"]["xy"][i])
            xy23.append(e["g"]["23"]["xy"][i])
        xy12, xy23 = np.array(xy12), np.array(xy23)
        print(
            f"  {cm:5.1f} {len(xy12):4d} {np.median(xy12) * 100:13.3f}cm {np.percentile(xy12, 90) * 100:6.3f}cm"
            f" {np.median(xy23) * 100:13.3f}cm {np.percentile(xy23, 90) * 100:6.3f}cm"
        )


def m2_failures(seed, cms):
    print("\n=== M2: where the lost successes go (FAILURES, share of each mode) ===")
    print(
        f"  {'cm':>5s} {'n':>5s} {'c2 never moved':>15s} {'c2 not stacked':>15s}"
        f" {'c3 never moved':>15s} {'c3 not stacked':>15s} {'geom ok, jaw no':>16s}"
    )
    for cm in cms:
        eps = load(path(seed, cm, failed=True))
        n = len(eps)
        counts = dict.fromkeys(["c2_still", "c2_bad", "c3_still", "c3_bad", "jaw"], 0)
        for e in eps:
            c1, c2, c3 = e["pos"]
            moved2 = np.linalg.norm(c2[-1] - c2[0]) > 0.02
            moved3 = np.linalg.norm(c3[-1] - c3[0]) > 0.02
            ok12_end = bool(e["g"]["12"]["ok"][-1])
            if e["g"]["both"].any() and not (e["g"]["both"] & (e["jaw"] <= GRIP_TOL)).any():
                counts["jaw"] += 1
            elif not moved2:
                counts["c2_still"] += 1
            elif not ok12_end:
                counts["c2_bad"] += 1
            elif not moved3:
                counts["c3_still"] += 1
            else:
                counts["c3_bad"] += 1
        print(
            f"  {cm:5.1f} {n:5d} {counts['c2_still'] / n * 100:14.1f}% {counts['c2_bad'] / n * 100:14.1f}%"
            f" {counts['c3_still'] / n * 100:14.1f}% {counts['c3_bad'] / n * 100:14.1f}%"
            f" {counts['jaw'] / n * 100:15.1f}%"
        )


def m3_loophole(seed, cms):
    print("\n=== M3: is the in-flight loophole helping the arc group? (SUCCESSES) ===")
    print(f"  {'cm':>5s} {'n':>5s} {'ends broken':>12s} {'cube speed at accept (med)':>28s} {'>0.05 m/s':>10s}")
    for cm in cms:
        eps = load(path(seed, cm, failed=False))
        broken, speeds = [], []
        for e in eps:
            q = e["g"]["both"] & (e["jaw"] <= GRIP_TOL)
            if not q.any():
                continue
            broken.append(not bool(e["g"]["both"][-1]))
            speeds.append(e["spd"][q].min())
        broken, speeds = np.array(broken), np.array(speeds)
        print(
            f"  {cm:5.1f} {len(broken):5d} {broken.mean() * 100:11.2f}% {np.median(speeds):27.4f}"
            f" {(speeds > 0.05).mean() * 100:9.1f}%"
        )


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cms = [float(x) for x in sys.argv[2:]] or [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    print(f"seed={seed}  amplitudes={cms}")
    m1_placement(seed, cms)
    m2_failures(seed, cms)
    m3_loophole(seed, cms)


if __name__ == "__main__":
    main()
