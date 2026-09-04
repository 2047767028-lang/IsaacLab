"""Where does the cube's off-centre seat come from: the grasp, or slipping during the carry?

Slipping in the jaws is an inertial-force question -- it needs acceleration beyond what friction
holds -- not a displacement question. So measure, per episode:
  seat at grasp    eef-to-cube_2 xy offset a few frames after the first jaw closure (fingers settled)
  seat at release  the same offset at the first jaw opening
  carry slip       release minus grasp: what changed while the cube was carried
  carry accel      peak and p90 of |d2 eef/dt2| while carrying (cm/frame^2), against the source demos
  in-hand yaw drift  change of cube_2 yaw relative to eef yaw between grasp and release (deg)

usage: seat_grasp_vs_release.py <out_dir> <tag> [<tag> ...]
"""
import os
import sys

import h5py
import numpy as np

SRC = "/home/pk/IsaacLab/datasets/annotated_dataset.hdf5"
OPEN_VAL = 0.04
SETTLE = 4


def transitions(grip):
    jaw = np.minimum(grip[:, 0], -grip[:, 1])
    closed = jaw < OPEN_VAL - 0.001
    return np.where(np.diff(closed.astype(int)) != 0)[0] + 1


def yaw(q):  # w,x,y,z
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def stats(d):
    pos, quat, grip = d["obs/eef_pos"][:], d["obs/eef_quat"][:], d["obs/gripper_pos"][:]
    c2 = d["states/rigid_object/cube_2/root_pose"][:]
    ev = transitions(grip)
    if len(ev) < 2:
        return None
    g, r = int(ev[0]), int(ev[1])
    gs = min(g + SETTLE, r - 1)
    seat_g = np.linalg.norm((pos[gs] - c2[gs, :3])[:2]) * 100
    seat_r = np.linalg.norm((pos[r] - c2[r, :3])[:2]) * 100
    acc = np.linalg.norm(np.diff(pos[gs:r], n=2, axis=0), axis=1) * 100 if r - gs > 4 else np.array([np.nan])
    yaw_g = yaw(c2[gs, 3:7]) - yaw(quat[gs])
    yaw_r = yaw(c2[r, 3:7]) - yaw(quat[r])
    dyaw = np.degrees(np.angle(np.exp(1j * (yaw_r - yaw_g))))
    return seat_g, seat_r, seat_r - seat_g, np.nanmax(acc), np.nanpercentile(acc, 90), abs(dyaw)


def q(v):
    v = np.asarray(v, float)
    return f"{np.nanmedian(v):5.2f}/{np.nanpercentile(v, 90):5.2f}"


def main(out, tags):
    print(f"  {'run':<10s} {'n':>4s} {'seat@grasp':>12s} {'seat@release':>13s} {'carry slip':>12s} "
          f"{'carry accel max':>16s} {'accel p90':>10s} {'in-hand yaw drift':>18s}")
    rows = []
    with h5py.File(SRC, "r") as f:
        for k in f["data"]:
            s = stats(f["data"][k])
            if s:
                rows.append(s)
    r = np.array(rows)
    print(f"  {'source':<10s} {len(r):4d} {q(r[:, 0]):>12s} {q(r[:, 1]):>13s} {q(r[:, 2]):>12s} "
          f"{q(r[:, 3]):>16s} {q(r[:, 4]):>10s} {q(r[:, 5]):>18s}")
    for tag in tags:
        rows = []
        for suffix in ("", "_failed"):
            p = os.path.join(out, f"ch_{tag}{suffix}.hdf5")
            if not os.path.exists(p):
                continue
            with h5py.File(p, "r") as f:
                for k in f["data"]:
                    s = stats(f["data"][k])
                    if s:
                        rows.append(s)
        if not rows:
            continue
        r = np.array(rows)
        print(f"  {tag:<10s} {len(r):4d} {q(r[:, 0]):>12s} {q(r[:, 1]):>13s} {q(r[:, 2]):>12s} "
              f"{q(r[:, 3]):>16s} {q(r[:, 4]):>10s} {q(r[:, 5]):>18s}")
    print("  (cm, cm/frame^2, deg; median/p90; seat = eef-to-cube_2 xy offset)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
