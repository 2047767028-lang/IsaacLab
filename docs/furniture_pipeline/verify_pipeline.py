"""Verify the OBJ -> USD -> Isaac Lab pipeline on the FurnitureBench square table.

Spawns the converted tabletop and four legs as rigid bodies, drops them onto a
ground plane, and steps physics. Passing means the SDF collision meshes load,
the bodies settle instead of exploding or sinking through the floor, and no
state goes non-finite.

Results are written to a file because Kit hijacks stdout.

Usage: isaaclab-python verify_pipeline.py [--assets DIR] [--report PATH]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify furniture asset pipeline in Isaac Lab.")
parser.add_argument("--assets", type=str, default="/home/pk/furniture_assembly/assets/square_table")
parser.add_argument("--report", type=str, default="/home/pk/furniture_assembly/pipeline_report.txt")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument(
    "--yup-fix",
    action="store_true",
    help="Rotate parts +90 deg about X. FurnitureBench meshes are authored Y-up "
    "(Isaac Gym convention) while the USD stage is Z-up, so without this the "
    "tabletop stands on its edge.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

# name -> (drop position). Legs are spread around the top, as in a real
# "parts laid out on the table" start state.
PARTS = {
    "square_table_top": (0.0, 0.0, 0.25),
    "square_table_leg1": (0.25, 0.15, 0.25),
    "square_table_leg2": (0.25, -0.15, 0.25),
    "square_table_leg3": (-0.25, 0.15, 0.25),
    "square_table_leg4": (-0.25, -0.15, 0.25),
}

report_lines = []


def log(msg):
    report_lines.append(str(msg))


def design_scene(assets_dir):
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    dome = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    dome.func("/World/Light", dome)

    entities = {}
    for name, pos in PARTS.items():
        usd_path = os.path.join(assets_dir, f"{name}.usd")
        if not os.path.isfile(usd_path):
            log(f"MISSING: {usd_path}")
            continue
        # (w, x, y, z) for +90 deg about X, which takes mesh Y-up to stage Z-up.
        rot = (0.70710678, 0.70710678, 0.0, 0.0) if args_cli.yup_fix else (1.0, 0.0, 0.0, 0.0)
        cfg = RigidObjectCfg(
            prim_path=f"/World/{name}",
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        )
        entities[name] = RigidObject(cfg=cfg)
        log(f"spawned {name} from {os.path.basename(usd_path)} at {pos}")
    return entities


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.0, 1.0, 0.6], target=[0.0, 0.0, 0.1])

    entities = design_scene(args_cli.assets)
    sim.reset()
    log(f"\nsimulation device: {sim.device}, dt={sim.get_physics_dt()}")
    log(f"loaded {len(entities)} parts\n")

    sim_dt = sim.get_physics_dt()
    checkpoints = {0, 60, 120, 240, args_cli.steps - 1}
    history = {name: [] for name in entities}

    for step in range(args_cli.steps):
        for obj in entities.values():
            obj.write_data_to_sim()
        sim.step()
        for obj in entities.values():
            obj.update(sim_dt)

        if step in checkpoints:
            log(f"--- step {step} (t={step*sim_dt:.3f}s) ---")
            for name, obj in entities.items():
                p = obj.data.root_pos_w[0]
                v = obj.data.root_lin_vel_w[0]
                log(f"  {name:<20} pos=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f})  "
                    f"|v|={torch.linalg.norm(v).item():.5f}")
        for name, obj in entities.items():
            history[name].append(obj.data.root_pos_w[0].clone())

    # verdict
    log("\n=== VERDICT ===")
    ok = True
    for name, obj in entities.items():
        p = obj.data.root_pos_w[0]
        v = obj.data.root_lin_vel_w[0]
        finite = bool(torch.isfinite(p).all() and torch.isfinite(v).all())
        speed = torch.linalg.norm(v).item()
        settled = speed < 0.02
        above_floor = p[2].item() > -0.02
        # drift over the last 60 steps tells us it really came to rest
        drift = torch.linalg.norm(history[name][-1] - history[name][-60]).item()
        status = "PASS" if (finite and settled and above_floor and drift < 0.01) else "FAIL"
        if status == "FAIL":
            ok = False
        log(f"  [{status}] {name:<20} z={p[2]:+.4f}m |v|={speed:.5f} "
            f"drift60={drift:.5f} finite={finite} above_floor={above_floor}")
    log(f"\nPIPELINE: {'PASS' if ok else 'FAIL'}")

    with open(args_cli.report, "w") as f:
        f.write("\n".join(report_lines) + "\n")


main()
simulation_app.close()
