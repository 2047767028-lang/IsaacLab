"""Execute the PR branches' actual source files, not a re-implementation of them.

The closed-loop runs earlier validated the *logic* of both fixes, but they did so by injecting
equivalent code into the installed 2.3.2 tree. The files that will be reviewed upstream -- the
develop-based `terminations.py` and `generation.py` -- were never executed. This loads those exact
files by path and drives them with mock environments, alongside the untouched develop versions so
the defect and the fix are demonstrated on the same inputs.

Needs the Omniverse app for `isaaclab.managers` imports, same as every test in the repository.

Usage:
  python exec_pr_code.py <worktree-with-fix/cubes-stacked-at-rest> <worktree-with-fix/mimic-max-num-failures>
"""

import sys

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import asyncio
import importlib.util
import os
import subprocess
import tempfile
from types import SimpleNamespace

import torch

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_show_to_temp(repo, ref, relpath):
    """Materialise <ref>:<relpath> into a temp file so the stock version can be loaded next to the fix."""
    src = subprocess.check_output(["git", "-C", repo, "show", f"{ref}:{relpath}"], text=True)
    fd, tmp = tempfile.mkstemp(suffix="_" + os.path.basename(relpath))
    with os.fdopen(fd, "w") as f:
        f.write(src)
    return tmp


# --------------------------------------------------------------------------------------------------
# Part 1: cubes_stacked
# --------------------------------------------------------------------------------------------------


class Arr:
    """Stand-in for develop's ProxyArray: exposes the tensor through `.torch`."""

    def __init__(self, t):
        self.torch = t


def make_env(n, cube_pos, cube_vel, jaw=(0.04, 0.04)):
    """cube_pos/cube_vel: dict name -> (n,3) tensors. Robot jaws at `jaw` for every env."""
    scene = {}
    for name in cube_pos:
        scene[name] = SimpleNamespace(data=SimpleNamespace(root_pos_w=Arr(cube_pos[name]), root_lin_vel_w=Arr(cube_vel[name])))
    joint_pos = torch.zeros(n, 9)
    joint_pos[:, 7] = jaw[0]
    joint_pos[:, 8] = jaw[1]
    robot = SimpleNamespace(
        find_joints=lambda names: ([7, 8], ["panda_finger_joint1", "panda_finger_joint2"]),
        data=SimpleNamespace(joint_pos=Arr(joint_pos)),
    )
    scene["robot"] = robot

    class Scene(dict):
        pass  # no `surface_grippers` attribute -> takes the parallel-gripper branch

    s = Scene(scene)
    cfg = SimpleNamespace(gripper_joint_names=["panda_finger_.*"], gripper_open_val=0.04)
    return SimpleNamespace(scene=s, cfg=cfg, device="cpu")


def stacked_scene(n, top_speed=0.0, bottom_speed=0.0, mid_speed=0.0):
    """A perfect three-cube stack: cube_1 on the table, cube_2 on it, cube_3 on that."""
    base = torch.tensor([0.5, 0.0, 0.0203]).repeat(n, 1)
    pos = {
        "cube_1": base.clone(),
        "cube_2": base + torch.tensor([0.0, 0.0, 0.0468]),
        "cube_3": base + torch.tensor([0.0, 0.0, 2 * 0.0468]),
    }
    vel = {
        "cube_1": torch.tensor([bottom_speed, 0.0, 0.0]).repeat(n, 1),
        "cube_2": torch.tensor([0.0, mid_speed, 0.0]).repeat(n, 1),
        "cube_3": torch.tensor([0.0, 0.0, -top_speed]).repeat(n, 1),  # falling straight down
    }
    return pos, vel


def part1(fix_repo, sc):
    print("\n=== cubes_stacked: stock develop vs PR branch ===")
    rel = "source/isaaclab_tasks/isaaclab_tasks/contrib/stack/mdp/terminations.py"
    stock = load_module(git_show_to_temp(fix_repo, "origin/develop", rel), "term_stock")
    fixed = load_module(git_show_to_temp(fix_repo, "fix/cubes-stacked-at-rest", rel), "term_fixed")

    # 1. resting stack passes both
    pos, vel = stacked_scene(1)
    env = make_env(1, pos, vel)
    check("stock: resting stack -> True", bool(stock.cubes_stacked(env)[0]))
    check("fixed: resting stack -> True", bool(fixed.cubes_stacked(env)[0]))

    # 2. the defect: top cube falling through the stacked configuration at 0.1 m/s
    pos, vel = stacked_scene(1, top_speed=0.10)
    env = make_env(1, pos, vel)
    check("stock: cube_3 falling at 0.10 m/s -> True  (the defect)", bool(stock.cubes_stacked(env)[0]))
    check("fixed: cube_3 falling at 0.10 m/s -> False", not bool(fixed.cubes_stacked(env)[0]))

    # 3. opt-out restores stock behaviour
    check("fixed: max_lin_vel=None -> True (opt-out)", bool(fixed.cubes_stacked(env, max_lin_vel=None)[0]))

    # 4. every cube in the checked set is subject to the test
    pos, vel = stacked_scene(1, mid_speed=0.10)
    check("fixed: cube_2 moving -> False", not bool(fixed.cubes_stacked(make_env(1, pos, vel))[0]))
    pos, vel = stacked_scene(1, bottom_speed=0.10)
    check("fixed: cube_1 moving -> False", not bool(fixed.cubes_stacked(make_env(1, pos, vel))[0]))

    # 5. threshold edges: solver jitter (0.03) passes, just over the default (0.051) fails
    pos, vel = stacked_scene(1, top_speed=0.030)
    check("fixed: 0.030 m/s (measured rest jitter) -> True", bool(fixed.cubes_stacked(make_env(1, pos, vel))[0]))
    pos, vel = stacked_scene(1, top_speed=0.051)
    check("fixed: 0.051 m/s -> False", not bool(fixed.cubes_stacked(make_env(1, pos, vel))[0]))
    pos, vel = stacked_scene(1, top_speed=0.049)
    check("fixed: 0.049 m/s -> True", bool(fixed.cubes_stacked(make_env(1, pos, vel))[0]))

    # 6. two-cube variant (cube_3_cfg=None): the third scene object is not consulted
    pos, vel = stacked_scene(1, top_speed=0.5)  # cube_3 moving fast, but excluded below
    env = make_env(1, pos, vel)
    r = fixed.cubes_stacked(env, cube_1_cfg=sc("cube_1"), cube_2_cfg=sc("cube_2"), cube_3_cfg=None)
    check("fixed: cube_3_cfg=None ignores a moving cube_3 -> True", bool(r[0]))
    pos, vel = stacked_scene(1, mid_speed=0.5)
    env = make_env(1, pos, vel)
    r = fixed.cubes_stacked(env, cube_1_cfg=sc("cube_1"), cube_2_cfg=sc("cube_2"), cube_3_cfg=None)
    check("fixed: cube_3_cfg=None still checks cube_2 -> False", not bool(r[0]))

    # 7. role remapping as the Franka variants do it: cube_1_cfg=cube_2, cube_2_cfg=cube_3, cube_3_cfg=None
    pos, vel = stacked_scene(1, top_speed=0.10)
    env = make_env(1, pos, vel)
    r = fixed.cubes_stacked(env, cube_1_cfg=sc("cube_2"), cube_2_cfg=sc("cube_3"), cube_3_cfg=None)
    check("fixed: remapped (cube_2,cube_3) with cube_3 falling -> False", not bool(r[0]))

    # 8. batched: per-env verdicts are independent
    n = 4
    pos, vel = stacked_scene(n)
    vel["cube_3"][1, 2] = -0.2  # env 1 falling
    vel["cube_2"][3, 0] = 0.2  # env 3 mid cube sliding
    r = fixed.cubes_stacked(make_env(n, pos, vel))
    check("fixed: batched verdict [T,F,T,F]", r.tolist() == [True, False, True, False], str(r.tolist()))
    r = stock.cubes_stacked(make_env(n, pos, vel))
    check("stock: same batch -> all True (the defect, batched)", r.tolist() == [True] * n, str(r.tolist()))

    # 9. gripper still gates success with the fix in place
    pos, vel = stacked_scene(1)
    env = make_env(1, pos, vel, jaw=(0.02, 0.02))
    check("fixed: closed gripper -> False (gripper check intact)", not bool(fixed.cubes_stacked(env)[0]))


# --------------------------------------------------------------------------------------------------
# Part 2: env_loop
# --------------------------------------------------------------------------------------------------


class Fuse(Exception):
    pass


def make_loop_env(gen_mod, outcomes, action_queue, fuse_after):
    """A fake env whose step() consumes one scripted attempt outcome per step and refills the queue.

    `outcomes` is an iterator of True/False (success/failure). The fuse raises after `fuse_after`
    steps so an unbounded loop cannot hang the test.
    """
    state = {"steps": 0}

    def step(actions):
        state["steps"] += 1
        if state["steps"] > fuse_after:
            raise Fuse(f"loop still running after {fuse_after} steps")
        try:
            ok = next(outcomes)
        except StopIteration:
            raise Fuse("scripted outcomes exhausted")
        if ok:
            gen_mod.num_success += 1
        else:
            gen_mod.num_failures += 1
        gen_mod.num_attempts += 1
        action_queue.put_nowait((0, torch.zeros(7)))

    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        action_space=SimpleNamespace(shape=(1, 7)),
        step=step,
        reset=lambda env_ids=None: None,
        sim=SimpleNamespace(is_stopped=lambda: False),
        cfg=SimpleNamespace(datagen_config=SimpleNamespace(generation_guarantee=True, generation_num_trials=3, max_num_failures=None)),
        _state=state,
    )


def run_loop(env_loop, gen_mod, outcomes, max_num_failures, num_trials=3, fuse_after=40):
    gen_mod.num_success = gen_mod.num_failures = gen_mod.num_attempts = 0
    loop = asyncio.get_event_loop()
    aq, rq = asyncio.Queue(), asyncio.Queue()
    aq.put_nowait((0, torch.zeros(7)))
    env = make_loop_env(gen_mod, iter(outcomes), aq, fuse_after)
    env.cfg.datagen_config.max_num_failures = max_num_failures
    env.cfg.datagen_config.generation_num_trials = num_trials
    try:
        env_loop(env, rq, aq, None, loop)
        return "exited", env._state["steps"]
    except Fuse as e:
        return "fuse", env._state["steps"]


def part2(fix_repo):
    print("\n=== env_loop: stock develop vs PR branch ===")
    rel = "source/isaaclab_mimic/isaaclab_mimic/datagen/generation.py"
    stock = load_module(git_show_to_temp(fix_repo, "origin/develop", rel), "gen_stock")
    fixed = load_module(git_show_to_temp(fix_repo, "fix/mimic-max-num-failures", rel), "gen_fixed")

    # Stock: cap is ignored. 25 straight failures, max_num_failures=5 -> runs into the fuse.
    all_fail = [False] * 100
    how, steps = run_loop(stock.env_loop, stock, all_fail, max_num_failures=5)
    check("stock: max_num_failures=5, all failing -> never stops (fuse)", how == "fuse", f"{how} after {steps} steps")

    # Fixed: same run stops at the cap.
    how, steps = run_loop(fixed.env_loop, fixed, all_fail, max_num_failures=5)
    check("fixed: max_num_failures=5, all failing -> exits", how == "exited", f"{how} after {steps} steps")
    check("fixed: stopped with exactly 5 failures", fixed.num_failures == 5, f"num_failures={fixed.num_failures}")

    # Fixed, default None: unbounded, exactly as before.
    how, steps = run_loop(fixed.env_loop, fixed, all_fail, max_num_failures=None)
    check("fixed: max_num_failures=None, all failing -> unbounded like stock (fuse)", how == "fuse", f"{how} after {steps} steps")

    # Fixed: successes still terminate first when they come first.
    mixed = [False, True, False, True, True] + [False] * 50
    how, steps = run_loop(fixed.env_loop, fixed, mixed, max_num_failures=100, num_trials=3)
    check("fixed: 3 successes before cap -> exits on successes", how == "exited" and fixed.num_success == 3, f"{how} succ={fixed.num_success} fail={fixed.num_failures}")

    # Fixed: cap reached before enough successes.
    mixed = [False, True, False, True, False, False] + [True] * 50
    how, steps = run_loop(fixed.env_loop, fixed, mixed, max_num_failures=4, num_trials=3)
    check("fixed: cap of 4 hit at 2 successes -> exits on cap", how == "exited" and fixed.num_failures == 4 and fixed.num_success == 2, f"{how} succ={fixed.num_success} fail={fixed.num_failures}")

    # Fixed: guarantee=False path unchanged (attempts-based) and cap still applies.
    gen = fixed
    gen.num_success = gen.num_failures = gen.num_attempts = 0
    loop = asyncio.get_event_loop()
    aq, rq = asyncio.Queue(), asyncio.Queue()
    aq.put_nowait((0, torch.zeros(7)))
    env = make_loop_env(gen, iter([False] * 100), aq, 40)
    env.cfg.datagen_config.generation_guarantee = False
    env.cfg.datagen_config.generation_num_trials = 10
    env.cfg.datagen_config.max_num_failures = None
    try:
        fixed.env_loop(env, rq, aq, None, loop)
        check("fixed: guarantee=False stops on attempts (unchanged)", gen.num_attempts == 10, f"attempts={gen.num_attempts}")
    except Fuse:
        check("fixed: guarantee=False stops on attempts (unchanged)", False, "fuse")


def main():
    fix_at_rest_repo = fix_cap_repo = sys.argv[1]
    from isaaclab.managers import SceneEntityCfg

    part1(fix_at_rest_repo, SceneEntityCfg)
    part2(fix_cap_repo)
    print(f"\nSUMMARY: {len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print("  FAILED:", f)
    sys.stdout.flush()
    os._exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()
