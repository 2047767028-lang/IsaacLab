"""Same-scene pairing: how does each arc run do on scenes where the reference run succeeded vs
failed? If copying the parallel run's pose only helps when that run itself succeeded, the table
built from all reference episodes is half poison."""
import h5py
import numpy as np

OUT = "/home/pk/IsaacLab/datasets/arc_sweep_diagnosis_runs/contact_hold"
CUBES = ("cube_1", "cube_2", "cube_3")


def load(tag):
    d = {}
    for suffix, ok in (("", True), ("_failed", False)):
        with h5py.File(f"{OUT}/ch_{tag}{suffix}.hdf5", "r") as f:
            for k in f["data"]:
                ro = f["data"][k]["states"]["rigid_object"]
                key = tuple(np.round(np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]), 3))
                d[key] = ok
    return d


ref = load("ref_none")
for tag in ("arc_none", "arc_snap", "arc_hold", "ref_hold"):
    run = load(tag)
    common = [k for k in run if k in ref]
    ok_ref = [k for k in common if ref[k]]
    bad_ref = [k for k in common if not ref[k]]
    s_ok = sum(run[k] for k in ok_ref)
    s_bad = sum(run[k] for k in bad_ref)
    print(f"{tag:9s} paired={len(common):3d}  on ref-success scenes: {s_ok}/{len(ok_ref)} = {100*s_ok/max(len(ok_ref),1):.1f}%"
          f"   on ref-failure scenes: {s_bad}/{len(bad_ref)} = {100*s_bad/max(len(bad_ref),1):.1f}%")
