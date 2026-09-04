"""At the first RELEASE: is the cube itself (not just the gripper) off-centre / fast / rotated,
and does that separate placements that held from ones that slid off?"""
import h5py
import numpy as np

GEN = "/home/pk/IsaacLab/datasets/perturbation_sweep_v2_arc"
OPEN_VAL = 0.04
CUBES = ("cube_1", "cube_2", "cube_3")


def events(grip):
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = np.where(jaw < OPEN_VAL - 0.001)[0]
    if not len(closed):
        return None, None
    g = int(closed[0])
    grasped = jaw[g + 5] if g + 5 < len(jaw) else jaw[g]
    reopen = np.where(jaw[g + 5:] > grasped + 0.001)[0]
    r = int(g + 5 + reopen[0]) if len(reopen) else None
    return g, r


def yaw(q):  # w,x,y,z
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_diff_mod90(q1, q2):
    d = (yaw(q1) - yaw(q2)) % (np.pi / 2)
    return np.degrees(min(d, np.pi / 2 - d))


def key_of(ro):
    return tuple(np.round(np.concatenate([ro[c]["root_pose"][0, :3] for c in CUBES]), 6))


def load_run(seed, cm):
    tag = f"cm{cm:.1f}".replace(".", "p")
    out = {}
    for suffix in ("", "_failed"):
        with h5py.File(f"{GEN}/seed{seed}/generated_{tag}{suffix}.hdf5", "r") as f:
            d = f["data"]
            for k in d.keys():
                ro = d[k]["states"]["rigid_object"]
                out[key_of(ro)] = {
                    "pos": d[k]["obs/eef_pos"][:],
                    "grip": d[k]["obs/gripper_pos"][:],
                    "c1": ro["cube_1"]["root_pose"][:],
                    "c2": ro["cube_2"]["root_pose"][:],
                    "v2": ro["cube_2"]["root_velocity"][:, :3],
                }
    return out


def q(v):
    v = np.asarray(v, float)
    return f"med {np.median(v):5.2f} p90 {np.percentile(v, 90):5.2f} (n={len(v)})"


def d_of(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt((a.var() + b.var()) / 2) or 1e-12
    return (b.mean() - a.mean()) / pooled


for cm in (0.5, 3.0):
    run = load_run(1, cm)
    rows = []
    for ep in run.values():
        g, r = events(ep["grip"])
        if r is None:
            continue
        c1, c2 = ep["c1"][r], ep["c2"][r]
        rows.append({
            "placed": np.linalg.norm((ep["c2"][-1, :3] - ep["c1"][-1, :3])[:2]) < 0.02,
            "cube_xy": np.linalg.norm((c2[:3] - c1[:3])[:2]) * 100,
            "cube_zgap": (c2[2] - c1[2]) * 100 - 4.68,          # bottom of cube_2 above top of cube_1
            "grip_xy": np.linalg.norm((ep["pos"][r] - c1[:3])[:2]) * 100,
            "seat": np.linalg.norm((ep["pos"][r] - c2[:3])[:2]) * 100,   # cube off-centre in the jaws
            "yaw": yaw_diff_mod90(c2[3:7], c1[3:7]),
            "grip_speed": np.linalg.norm(ep["pos"][r] - ep["pos"][r - 1]) * 100,
            "cube_speed": np.linalg.norm(ep["v2"][r]) * 100,
        })
    placed = np.array([x["placed"] for x in rows])
    print(f"=== seed 1, arc {cm} cm: {len(rows)} releases, placed {placed.sum()} ({placed.mean()*100:.1f}%) ===")
    print(f"  {'quantity at release':<28s} {'placed':>26s} {'not placed':>26s}  d")
    for name, key in (
        ("cube2-cube1 xy offset (cm)", "cube_xy"),
        ("cube2 bottom above cube1 top", "cube_zgap"),
        ("gripper-cube1 xy offset", "grip_xy"),
        ("cube off-centre in jaws (xy)", "seat"),
        ("yaw misalignment mod90 (deg)", "yaw"),
        ("gripper speed (cm/frame)", "grip_speed"),
        ("cube2 speed (cm/s)", "cube_speed"),
    ):
        v = np.array([x[key] for x in rows])
        print(f"  {name:<28s} {q(v[placed]):>26s} {q(v[~placed]):>26s}  {d_of(v[placed], v[~placed]):+.2f}")
    print()
