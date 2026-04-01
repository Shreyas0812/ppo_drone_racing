# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL policy under a sweep of domain randomization parameters.

Each environment is assigned one DR configuration (a combination of twr_scale,
k_aero_scale, and pid_scale).  The script runs until every environment has
completed ``--eval_laps`` laps (or the simulation is closed), then prints a
summary table with per-lap times and the mean.
"""

"""Launch Isaac Sim Simulator first."""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

local_rsl_path = os.path.join(_project_root, "src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)
    print(f"[INFO] Using local rsl_rl from: {local_rsl_path}")
else:
    print(f"[WARNING] Local rsl_rl not found at: {local_rsl_path}")

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate RSL-RL policy under DR sweep.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=2500, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (auto-set to grid size if None).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--eval_laps", type=int, default=3, help="Number of laps to collect per DR config.")
parser.add_argument("--max_steps", type=int, default=100000, help="Safety cap: stop after this many steps regardless.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import itertools

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# Import extensions to set up environment tasks
import src.isaac_quad_sim2real.tasks   # noqa: F401


# ---------------------------------------------------------------------------
# DR sweep grid — edit these to taste
# ---------------------------------------------------------------------------
TWR_SCALES    = [0.95, 1.00, 1.05]          # ±5 % thrust-to-weight
K_AERO_SCALES = [0.50, 1.00, 2.00]          # 50 % / nominal / 200 % aero drag
PID_SCALES    = [0.85, 1.00, 1.15]          # ±15 % PID gains (roll/pitch/yaw kp/ki/kd)
# ---------------------------------------------------------------------------


def build_dr_configs():
    """Return list of (twr_scale, k_aero_scale, pid_scale) tuples — the full grid."""
    return list(itertools.product(TWR_SCALES, K_AERO_SCALES, PID_SCALES))


def apply_dr_configs(raw_env, configs):
    """Write per-env DR parameters directly into the physics tensors.

    Parameters are set once after env creation.  reset_idx() does NOT overwrite
    these tensors (it only resets pose/velocity/action buffers), so the assigned
    values persist for the entire evaluation run.
    """
    cfg = raw_env.cfg
    n   = raw_env.num_envs

    for i, (twr_s, aero_s, pid_s) in enumerate(configs[:n]):
        # Thrust-to-weight
        raw_env._thrust_to_weight[i] = cfg.thrust_to_weight * twr_s

        # Aerodynamic drag
        raw_env._K_aero[i, :2] = cfg.k_aero_xy * aero_s
        raw_env._K_aero[i,  2] = cfg.k_aero_z  * aero_s

        # PID gains — roll/pitch
        raw_env._kp_omega[i, :2] = cfg.kp_omega_rp * pid_s
        raw_env._ki_omega[i, :2] = cfg.ki_omega_rp * pid_s
        raw_env._kd_omega[i, :2] = cfg.kd_omega_rp * pid_s

        # PID gains — yaw
        raw_env._kp_omega[i, 2] = cfg.kp_omega_y * pid_s
        raw_env._ki_omega[i, 2] = cfg.ki_omega_y * pid_s
        raw_env._kd_omega[i, 2] = cfg.kd_omega_y * pid_s

    print(f"[EVAL] Applied {min(len(configs), n)} DR configs to {n} environments.")


def print_results(configs, lap_times, gate_pass_times, eval_laps):
    """Print a formatted summary table."""
    header_laps = "  ".join(f"Lap{k+1:>1}" for k in range(eval_laps))
    print("\n" + "=" * 90)
    print(f"{'Config':<42}  {header_laps}  {'Mean':>7}  {'Laps':>4}")
    print("-" * 90)

    for i, (twr_s, aero_s, pid_s) in enumerate(configs):
        lts = lap_times[i][:eval_laps]
        label = f"twr={twr_s:.2f}  aero={aero_s:.2f}  pid={pid_s:.2f}"

        lap_cols = []
        for k in range(eval_laps):
            lap_cols.append(f"{lts[k]:5.2f}s" if k < len(lts) else "  DNF ")

        mean_str = f"{sum(lts)/len(lts):5.2f}s" if lts else "  DNF "
        laps_str = f"{len(lts)}/{eval_laps}"
        print(f"{label:<42}  {'  '.join(lap_cols)}  {mean_str}  {laps_str:>4}")

    print("=" * 90)

    # Per-gate crossing times for nominal config (env whose config is all-ones)
    try:
        nom_idx = next(
            i for i, (twr_s, aero_s, pid_s) in enumerate(configs)
            if abs(twr_s - 1.0) < 1e-6 and abs(aero_s - 1.0) < 1e-6 and abs(pid_s - 1.0) < 1e-6
        )
        print(f"\nGate crossing times for nominal config (env {nom_idx}):")
        for gate_idx, t in gate_pass_times[nom_idx]:
            print(f"  Gate {gate_idx} passed at t={t:.3f}s")
    except StopIteration:
        pass


def main():
    """Evaluate policy over a DR parameter sweep."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    configs = build_dr_configs()
    n_configs = len(configs)

    # Override num_envs to match grid size unless user specified it explicitly
    if args_cli.num_envs is None:
        env_cfg.scene.num_envs = n_configs
        print(f"[EVAL] DR grid has {n_configs} configs — setting num_envs={n_configs}.")
    else:
        if args_cli.num_envs < n_configs:
            configs = configs[:args_cli.num_envs]
            n_configs = args_cli.num_envs
            print(f"[EVAL] --num_envs {args_cli.num_envs} < grid size; truncating to {n_configs} configs.")
        elif args_cli.num_envs > n_configs:
            # Repeat configs to fill all envs
            configs = (configs * ((args_cli.num_envs // n_configs) + 1))[:args_cli.num_envs]
            n_configs = args_cli.num_envs
            print(f"[EVAL] --num_envs {args_cli.num_envs} > grid size; repeating configs.")

    # Load checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    env_cfg.is_train = False
    env_cfg.max_motor_noise_std = 0.0
    env_cfg.seed = args_cli.seed

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        from isaaclab.utils.dict import print_dict
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer,
        path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir, filename="policy.onnx"
    )

    policy  = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    raw_env = env.unwrapped  # QuadcopterEnv

    # Apply DR configs before the first step (persists across resets in play mode)
    apply_dr_configs(raw_env, configs)

    n_gates   = raw_env._waypoints.shape[0]
    dt        = raw_env.cfg.sim.dt * raw_env.cfg.decimation

    # Per-env tracking state
    lap_times       = [[] for _ in range(n_configs)]   # list of lap durations per env
    gate_pass_times = [[] for _ in range(n_configs)]   # (gate_idx, sim_time) per env
    lap_start_time  = torch.zeros(n_configs, device=raw_env.device)
    sim_time        = 0.0

    # reset environment
    obs = env.get_observations()
    if hasattr(obs, "get"):
        obs = obs["policy"]

    prev_wp    = raw_env._idx_wp.clone()
    prev_gates = raw_env._n_gates_passed.clone()
    step       = 0

    print(f"\n[EVAL] Starting evaluation: {n_configs} configs, {args_cli.eval_laps} laps each.")
    print(f"[EVAL] Max steps: {args_cli.max_steps}.  dt={dt*1000:.2f}ms/step.\n")

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if hasattr(obs, "get"):
                obs = obs["policy"]

        sim_time += dt
        step     += 1

        curr_wp    = raw_env._idx_wp
        curr_gates = raw_env._n_gates_passed

        for i in range(n_configs):
            # Gate crossing: waypoint index changed
            if curr_wp[i] != prev_wp[i]:
                gate_pass_times[i].append((int(prev_wp[i]), sim_time))

            # Lap complete: _n_gates_passed wrapped back to 0 after being >0
            # Also fires when env resets mid-lap (prev_gates[i] > 0 but env died) —
            # only count if all n_gates gates were actually passed this lap.
            if prev_gates[i] > 0 and curr_gates[i] == 0:
                # Count gate crossings since last lap_start to verify full lap
                gates_this_lap = sum(
                    1 for _, t in gate_pass_times[i]
                    if t > float(lap_start_time[i])
                )
                if gates_this_lap >= n_gates:
                    lap_duration = sim_time - float(lap_start_time[i])
                    lap_times[i].append(lap_duration)
                    completed = len(lap_times[i])
                    print(f"  [env {i:3d}] Lap {completed} done  {lap_duration:.3f}s"
                          f"  (twr={configs[i][0]:.2f} aero={configs[i][1]:.2f} pid={configs[i][2]:.2f})")
                lap_start_time[i] = sim_time

        prev_wp    = curr_wp.clone()
        prev_gates = curr_gates.clone()

        # Stop once all envs have the required number of laps (or hit the step cap)
        finished = all(len(lap_times[i]) >= args_cli.eval_laps for i in range(n_configs))
        if finished or step >= args_cli.max_steps:
            if finished:
                print("\n[EVAL] All environments completed the required laps.")
            else:
                print(f"\n[EVAL] Reached max_steps={args_cli.max_steps}. Printing partial results.")
            break

    print_results(configs, lap_times, gate_pass_times, args_cli.eval_laps)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()



# python scripts/rsl_rl/play_race_evaluate.py \
#   --task Isaac-Quadcopter-Race-v0 \
#   --load_run 2026-04-01_02-44-57 \
#   --checkpoint model_15000_4224.pt \
#   --eval_laps 3 \
#   --headless