# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

        # Additional parameters for reward and observation calculations
        self._yaw_diff = torch.zeros(self.num_envs, device=self.device)
        # self._last_gate_x = torch.zeros(self.num_envs, device=self.device)
        # self._powerloop_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lap_start_time = torch.zeros(self.num_envs, device=self.device)
        self._powerloop_start_time = torch.zeros(self.num_envs, device=self.device)

        # Powerloop phase sequence tracking
        self._visited_p1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._visited_p2 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._visited_p3 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._powerloop_done_this_lap = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 3-lap race tracking (active after 5000 iterations)
        self._total_gates_passed = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self._race_start_time = torch.zeros(self.num_envs, device=self.device)
        self._had_crash = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. """

        # TODO ----- START ----- Define the tensors required for your custom reward structure
        
        ############################## Reward for passing through a gate and updating the waypoint and desired position accordingly ##############################
        # check to change waypoint
        x_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 0]
        y_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 1]
        z_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 2]
        
        # dist_to_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)
        # crossed_plane = (self.env._prev_x_drone_wrt_gate > 0) & (x_drone_wrt_gate <= 0)
        # y_pass_safely = torch.abs(y_drone_wrt_gate) < 0.4
        # z_pass_safely = torch.abs(z_drone_wrt_gate) < 0.4

        # gate_passed = crossed_plane & (dist_to_gate < 0.6)
        # gate_passed = (dist_to_gate < 0.6)

        crossed_plane = (self.env._prev_x_drone_wrt_gate > 0) & (x_drone_wrt_gate <= 0)
        y_pass_safely = torch.abs(y_drone_wrt_gate) < 0.72
        z_pass_safely = torch.abs(z_drone_wrt_gate) < 0.72

        # Require minimum forward velocity through the gate to reject crash-bounce false positives
        # gate_rot_matrix_pass = matrix_from_quat(self.env._waypoints_quat[self.env._idx_wp])
        # drone_vel_w_pass = self.env._robot.data.root_com_lin_vel_w
        # vel_fwd = (drone_vel_w_pass * gate_rot_matrix_pass[:, :, 0]).sum(dim=1)
        # flying_through = vel_fwd > 0.3  # m/s — below this is a bounce, not a clean pass

        # gate_passed = crossed_plane & y_pass_safely & z_pass_safely & flying_through
        gate_passed = crossed_plane & y_pass_safely & z_pass_safely
        # Gate 3 only counts if the powerloop arc was executed: drone must have been above gate
        # height on the wrong side (Phase 2) earlier this episode. Without this, the drone can
        # pass gate 3 via a horizontal detour and never learn the vertical loop.
        at_gate3 = (self.env._idx_wp == 3)
        gate_passed = torch.where(at_gate3, gate_passed & self._visited_p2, gate_passed)
        gate3_passed = gate_passed & at_gate3  # capture before _idx_wp is incremented

        self.env._prev_x_drone_wrt_gate = x_drone_wrt_gate.clone()

        ids_gate_passed = torch.where(gate_passed)[0]
        self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]

        # Update n_gates_passed when gate is passed and reset the counter if a lap is completed
        self.env._n_gates_passed[ids_gate_passed] += 1
        lap_completed = self.env._n_gates_passed[ids_gate_passed] >= self.env._waypoints.shape[0]
        self.env._n_gates_passed[ids_gate_passed] = torch.where(lap_completed, 0, self.env._n_gates_passed[ids_gate_passed])
        self.env._yaw_n_laps[ids_gate_passed] += torch.where(lap_completed, 1, 0)

        lap_completed_all = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if len(ids_gate_passed) > 0:
            lap_completed_all[ids_gate_passed] = lap_completed

        # Lap time bonus: reward faster laps on top of the flat lap_complete bonus
        current_time = self.env.episode_length_buf * self.env.cfg.sim.dt * self.env.cfg.decimation
        lap_time = (current_time - self._lap_start_time).clamp(min=0.1)
        target_lap_time = getattr(self.env, 'rew', {}).get('target_lap_time', 5.0)
        lap_time_bonus = torch.exp(-lap_time / target_lap_time) * lap_completed_all.float()
        self._lap_start_time[lap_completed_all] = current_time[lap_completed_all]

        # set desired positions in the world frame
        self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
        self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]

        # Update gate-relative pose immediately so observations see the NEW gate this step
        if len(ids_gate_passed) > 0:
            self.env._pose_drone_wrt_gate[ids_gate_passed], _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
                self.env._robot.data.root_link_pos_w[ids_gate_passed]
            )
            # Refresh local variables so downstream reward logic sees the new gate
            x_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 0]
            y_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 1]
            z_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 2]

        ###################################################################################################################################################################

        ####################################### Reward for progress towards the goal (current gate) ##############################

        self.env._last_distance_to_goal[ids_gate_passed] = torch.linalg.norm(self.env._desired_pos_w[ids_gate_passed] - self.env._robot.data.root_link_pos_w[ids_gate_passed], dim=1)

        prev_distance_to_goal = self.env._last_distance_to_goal
        curr_distance_to_goal = torch.linalg.norm(self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1)

        # progress = prev_distance_to_goal - curr_distance_to_goal
        progress_norm_scale = getattr(self.env, 'rew', {}).get('progress_norm_scale', 0.05)

        progress = torch.tanh((prev_distance_to_goal - curr_distance_to_goal) / progress_norm_scale)

        # Remove Euclidean progress for gate 3 when on wrong side — it pulls the drone
        # toward gate 3 from y<0 without doing the loop.
        gate3_wrong_side_mask = (self.env._idx_wp == 3) & (x_drone_wrt_gate <= 0)
        progress = torch.where(gate3_wrong_side_mask, torch.zeros_like(progress), progress)

        self.env._last_distance_to_goal = curr_distance_to_goal.clone()

        # Velocity toward the next gate (world frame): dot(vel, unit_vec_to_gate), normalized to [0,1]
        to_gate_vec = self.env._desired_pos_w - self.env._robot.data.root_link_pos_w
        to_gate_dist = torch.linalg.norm(to_gate_vec, dim=1, keepdim=True).clamp(min=0.01)
        to_gate_dir = to_gate_vec / to_gate_dist
        max_vel_to_gate = getattr(self.env, 'rew', {}).get('max_vel_to_gate', 5.0)
        vel_to_gate_reward = ((self.env._robot.data.root_com_lin_vel_w * to_gate_dir).sum(dim=1) / max_vel_to_gate).clamp(0, 1)
        vel_to_gate_reward = torch.where(gate3_wrong_side_mask, torch.zeros_like(vel_to_gate_reward), vel_to_gate_reward)

        ###################################################################################################################################################################


        ###################################### Reward for passing through a gate in the right yaw ##############################

        yaw_angle_scale = getattr(self.env, 'rew', {}).get('yaw_angle_scale', 0.15)
        _, _, drone_yaw = euler_xyz_from_quat(self.env._robot.data.root_quat_w)
        yaw_diff = drone_yaw - self.env._waypoints[self.env._idx_wp, -1]
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
        yaw_reward = torch.exp(-yaw_diff**2 / (2 * (yaw_angle_scale**2)))  # Gaussian reward centered at 0 with std dev of 0.15 radians
        
        self._yaw_diff = yaw_diff.clone()  # Store for Observation

        ###################################### Reward for crashing (contact with environment) ##############################


        # compute crashed environments if contact detected for 100 timesteps
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # Track whether any crash occurred this episode (for 3-lap no-crash bonus)
        self._had_crash |= (crashed * mask).bool()
        # TODO ----- END -----

        ######################################## Dealing with Powerloop at Gate 3 ########################################

        # Wrong side penalty: penalize being front the gate and moving towards it

        gate_rot_matrix = matrix_from_quat(self.env._waypoints_quat[self.env._idx_wp])  # [num_envs, 3, 3]
        drone_vel_w = self.env._robot.data.root_com_lin_vel_w  # world frame [num_envs, 3]

        # Project world-frame velocity onto each gate axis (dot product is frame-invariant)
        vel_along_gate_x = (drone_vel_w * gate_rot_matrix[:, :, 0]).sum(dim=1)  # along gate x (pass-through direction)
        vel_along_gate_y = (drone_vel_w * gate_rot_matrix[:, :, 1]).sum(dim=1)  # along gate y (lateral)
        vel_along_gate_z = (drone_vel_w * gate_rot_matrix[:, :, 2]).sum(dim=1)  # along gate z (vertical in gate frame)

        approach_x = (x_drone_wrt_gate > 0).float()
        withdraw_x = (x_drone_wrt_gate <= 0).float()

        approach_y = (y_drone_wrt_gate > 0).float()
        withdraw_y = (y_drone_wrt_gate <= 0).float()

        # Threshold at 0.5m above gate center = gate top (gate_side=1.0m, so top is at z_drone_wrt_gate=0.5)
        approach_z = (z_drone_wrt_gate > 0.6).float()
        withdraw_z = (z_drone_wrt_gate <= 0.6).float()

        gate3 = (self.env._idx_wp == 3).float()

        max_vel_gate3 = getattr(self.env, 'rew', {}).get('max_vel_gate3', 5.0)

        if (self.env._idx_wp == 3).any():
            # Phase 1: wrong side, below gate height (withdraw_x + withdraw_z)
            # Drone needs to climb and not go deeper into wrong side
            p1 = gate3 * withdraw_x * withdraw_z
            p1_x_reward = p1 * (vel_along_gate_x / max_vel_gate3).clamp(0, 1)                               # toward gate in x (world +y) → reward
            p1_y_reward = p1 * (-vel_along_gate_y / max_vel_gate3).clamp(0, 1)                              # away from gate 2 in y (world +x) → reward
            p1_z_reward = p1 * (vel_along_gate_z / max_vel_gate3).clamp(0, 1)                               # climbing (world +z) → reward
            p1_penalty  = p1 * ((-vel_along_gate_z).clamp(min=0) + (-vel_along_gate_x).clamp(min=0)).clamp(max=max_vel_gate3) / max_vel_gate3  # sinking or going deeper → penalize

            # Phase 2: wrong side, above gate height (withdraw_x + approach_z)
            # Drone needs to arc over toward correct side
            p2 = gate3 * withdraw_x * approach_z
            p2_x_reward  = p2 * (vel_along_gate_x / max_vel_gate3).clamp(0, 1)   # arcing toward correct side (world +y) → reward
            p2_z_reward  = p2 * (-vel_along_gate_z / max_vel_gate3).clamp(0, 1)  # descending while arcing → reward (tightens loop)
            p2_z_penalty = p2 * (vel_along_gate_z / max_vel_gate3).clamp(0, 1)   # still climbing in p2 → penalize

            # Phase 3: correct side, above gate height (approach_x + approach_z)
            # Drone needs to descend toward gate and not fly back
            p3 = gate3 * approach_x * approach_z
            p3_x_reward = p3 * (-vel_along_gate_x / max_vel_gate3).clamp(0, 1)                              # toward gate in x (world -y) → reward
            p3_y_reward = p3 * (-y_drone_wrt_gate.sign() * vel_along_gate_y / max_vel_gate3).clamp(0, 1)   # centering in y → reward
            p3_z_reward = p3 * (-vel_along_gate_z / max_vel_gate3).clamp(0, 1)                              # descending (world -z) → reward
            p3_penalty  = p3 * (vel_along_gate_x / max_vel_gate3).clamp(0, 1)                               # flying back to wrong side → penalize

            # Accumulate phase visits — start powerloop timer on first entry into phase 1
            newly_entered_p1 = (p1 > 0) & ~self._visited_p1
            self._powerloop_start_time[newly_entered_p1] = current_time[newly_entered_p1]
            self._visited_p1 |= (p1 > 0)
            self._visited_p2 |= ((p2 > 0) & self._visited_p1)
            self._visited_p3 |= ((p3 > 0) & self._visited_p2)
        else:
            # same vars are 0 when no env is targeting gate 3
            p1_x_reward = p1_y_reward = p1_z_reward = p1_penalty = torch.zeros(self.num_envs, device=self.device)
            p2_x_reward = p2_z_reward = p2_z_penalty = torch.zeros(self.num_envs, device=self.device)
            p3_x_reward = p3_y_reward = p3_z_reward = p3_penalty = torch.zeros(self.num_envs, device=self.device)

        # Per-step penalty while targeting gate 3 — prevents indefinite phase farming
        gate3_time_penalty = (self.env._idx_wp == 3).float()

        # Sequence bonus: fires when gate 3 is passed having visited p1→p2→p3 in order.
        # Computed outside the if/else because _idx_wp is already incremented to 4 at gate pass time.
        powerloop_sequence = (gate3_passed & self._visited_p1 & self._visited_p2 & self._visited_p3).float()
        completed = powerloop_sequence.bool()
        self._visited_p1[completed] = False
        self._visited_p2[completed] = False
        self._visited_p3[completed] = False
        self._powerloop_done_this_lap |= completed

        # Gate lap_complete on powerloop having been done this lap; reset flag on lap completion
        lap_completed_all = lap_completed_all & self._powerloop_done_this_lap
        self._powerloop_done_this_lap[lap_completed_all] = False

        # Powerloop time bonus: exponential bonus for completing the loop quickly
        target_powerloop_time = getattr(self.env, 'rew', {}).get('target_powerloop_time', 2.5)
        powerloop_time = (current_time - self._powerloop_start_time).clamp(min=0.1)
        powerloop_time_bonus = torch.exp(-powerloop_time / target_powerloop_time) * powerloop_sequence

        # 3-lap race rewards (active after 5000 iterations)
        it = self.env.iteration if hasattr(self.env, 'iteration') else 0
        self._total_gates_passed[ids_gate_passed] += 1
        race_complete = (self._total_gates_passed == 22)  # 7 gates * 3 laps + 1 (gate 0 on return)

        no_crash_bonus = torch.zeros(self.num_envs, device=self.device)
        race_time_bonus = torch.zeros(self.num_envs, device=self.device)
        if it > 5000 and race_complete.any():
            target_race_time = getattr(self.env, 'rew', {}).get('target_race_time', 15.0)
            race_time = (current_time - self._race_start_time).clamp(min=0.1)
            race_time_bonus = race_complete.float() * torch.exp(-race_time / target_race_time)
            no_crash_bonus = race_complete.float() * (~self._had_crash).float()
            self._total_gates_passed[race_complete] = 0
            self._race_start_time[race_complete] = current_time[race_complete]
            self._had_crash[race_complete] = False

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "passing_gate": gate_passed.int() * self.env.rew['passing_gate_reward_scale'],
                "lap_complete": lap_completed_all.float() * self.env.rew['lap_complete_reward_scale'],
                "progress_goal": progress * self.env.rew['progress_goal_reward_scale'],
                "vel_to_gate": vel_to_gate_reward * self.env.rew['vel_to_gate_reward_scale'],
                "yaw": yaw_reward * self.env.rew['yaw_reward_scale'],
                "crash": crashed * self.env.rew['crash_reward_scale'],
                # Gate 3 powerloop phase rewards/penalties
                "p1_x_reward": p1_x_reward * self.env.rew['p1_x_reward_reward_scale'],  # Phase 1: toward gate in x
                "p1_y_reward": p1_y_reward * self.env.rew['p1_y_reward_reward_scale'],  # Phase 1: away from gate 2 in y
                "p1_z_reward": p1_z_reward * self.env.rew['p1_z_reward_reward_scale'],  # Phase 1: climbing
                "p1_penalty":  p1_penalty  * self.env.rew['p1_penalty_reward_scale'],   # Phase 1: sinking or going deeper
                "p2_x_reward":        p2_x_reward        * self.env.rew['p2_x_reward_reward_scale'],        # Phase 2: arc to correct side
                "p2_z_reward":      p2_z_reward      * self.env.rew['p2_z_reward_reward_scale'],      # Phase 2: descending while arcing
                "p2_z_penalty": p2_z_penalty * self.env.rew['p2_z_penalty_reward_scale'], # Phase 2: still climbing
                "p3_x_reward": p3_x_reward * self.env.rew['p3_x_reward_reward_scale'],  # Phase 3: toward gate in x
                "p3_y_reward": p3_y_reward * self.env.rew['p3_y_reward_reward_scale'],  # Phase 3: centering in y
                "p3_z_reward": p3_z_reward * self.env.rew['p3_z_reward_reward_scale'],  # Phase 3: descending
                "p3_penalty":  p3_penalty  * self.env.rew['p3_penalty_reward_scale'],   # Phase 3: flying back to wrong side
                "powerloop_sequence": powerloop_sequence * self.env.rew['powerloop_sequence_reward_scale'],  # Bonus for p1→p2→p3 sequence
                "powerloop_time_bonus": powerloop_time_bonus * self.env.rew['powerloop_time_bonus_reward_scale'],  # Exponential bonus for faster powerloop
                "gate3_time_penalty": gate3_time_penalty * self.env.rew['gate3_time_penalty_reward_scale'],  # Per-step cost while at gate 3
                "lap_time_bonus": lap_time_bonus * self.env.rew['lap_time_bonus_reward_scale'],
                "race_no_crash_bonus": no_crash_bonus * self.env.rew['race_no_crash_bonus_reward_scale'],
                "race_time_bonus": race_time_bonus * self.env.rew['race_time_bonus_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:   # This else condition implies eval is called with play_race.py. Can be useful to debug at test-time
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations. Read reset_idx() and quadcopter_env.py to see which drone info is extracted from the sim.
        The following code is an example. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define tensors for your observation space. Be careful with frame transformations
        #### Basic drone states, modify for your needs)
        drone_pose_w = self.env._robot.data.root_link_pos_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_quat_w = self.env._robot.data.root_quat_w

        ##### Some example observations you may want to explore using
        # Angular velocities (referred to as body rates)
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b  # [roll_rate, pitch_rate, yaw_rate]

        # Current target gate information
        current_gate_idx = self.env._idx_wp.unsqueeze(-1).float()       # [num_envs, 1]
        current_gate_pos_w = self.env._waypoints[self.env._idx_wp, :3]  # World position of current gate [num_envs, 3]
        current_gate_yaw = self.env._waypoints[self.env._idx_wp, -1].unsqueeze(-1)  # Yaw orientation [num_envs, 1]

        # Relative position to current gate in gate frame
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate

        # Relative position to current gate in body frame
        # gate_pos_b, _ = subtract_frame_transforms(
        #     self.env._robot.data.root_link_pos_w,
        #     self.env._robot.data.root_quat_w,
        #     current_gate_pos_w
        # )

        # Previous actions
        prev_actions = self.env._previous_actions  # Shape: (num_envs, 4)

        # Number of gates passed (normalized to [0, 1])
        gates_passed = self.env._n_gates_passed.unsqueeze(1).float() / self.env._waypoints.shape[0]

        # yaw difference to the gate
        yaw_diff = self._yaw_diff.unsqueeze(1)  # Shape: (num_envs, 1)

        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START ----- List your observation tensors here to be concatenated together
            [
                drone_lin_vel_b,    # velocity in the body frame (3 dims)
                drone_ang_vel_b,    # angular velocity in the body frame (3 dims)
                drone_quat_w,       # quaternion in the world frame (4 dims)
                drone_pos_gate_frame,
                gates_passed,       # number of gates passed (1 dim)
                yaw_diff,          # yaw difference to the gate (1 dim)
                prev_actions,       # previous actions (4 dims)
            ],
            # TODO ----- END -----
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # This example code initializes the drone 2m behind the first gate. You should delete it or heavily
        # modify it once you begin the racing task.

        # Staged curriculum: gradually unlock start positions as training progresses.
        # Gate 2 is intentionally skipped — it is naturally reached with momentum from gate 1.
        # Gate 3 is unlocked separately as it requires the powerloop height behavior.
        domain_randomization = False
        it = self.env.iteration if hasattr(self.env, 'iteration') else 0
        # if it < 1500:
        #     # Gate 0 only
        #     pool = [0]
        # elif it < 2500:
        #     # Gates 0, 1, and 3 (powerloop segment unlocked)
        #     pool = [0, 1, 3]
        # else:
        #     # All gates
        #     pool = list(range(self.env._waypoints.shape[0]))

        #     if it > 5000:
        #         # Start Domain Randomization after 5000 iterations (can be adjusted based on training progress)
        #         domain_randomization = True
        pool = [0]
        if it > 4000:
            domain_randomization = False
        elif it > 2000:
            # Start Domain Randomization after 2000 iterations (aligned with first randomization in _randomize_parameters)
            domain_randomization = True
        
        pool_tensor = torch.tensor(pool, device=self.device, dtype=self.env._idx_wp.dtype)
        waypoint_indices = pool_tensor[torch.randint(0, len(pool), (n_reset,), device=self.device)]

        # For crashed/timed-out envs: reset to the gate they were at.
        # Disabled after 5000 iterations so every episode is a clean 3-lap race from gate 0.
        if it <= 5000:
            crashed_mask = self.env.reset_terminated[env_ids]
            if crashed_mask.any():
                waypoint_indices = torch.where(crashed_mask, self.env._idx_wp[env_ids], waypoint_indices)

            timeout_mask = self.env.reset_time_outs[env_ids]
            if timeout_mask.any():
                waypoint_indices = torch.where(timeout_mask, self.env._idx_wp[env_ids], waypoint_indices)

        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        x_local = -2.0 * torch.ones(n_reset, device=self.device)
        y_local = torch.zeros(n_reset, device=self.device)

        # rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot

        at_gate0 = (waypoint_indices == 0)
        initial_z = torch.where(at_gate0, torch.full((n_reset,), 0.05, device=self.device), z_wp)

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # Forward momentum so powerloop is feasible right after spawn
        forward_speed = torch.empty(n_reset, device=self.device).uniform_(0.5, 1.0)
        default_root_state[:, 7] = -forward_speed * cos_theta # world frame velocity in x direction
        default_root_state[:, 8] = -forward_speed * sin_theta # world frame velocity in y direction
        default_root_state[:, 9] = 0.0  # velocity in z direction

        # point drone towards the zeroth gate
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + torch.empty(n_reset, device=self.device).uniform_(-0.15, 0.15)
        )
        default_root_state[:, 3:7] = quat
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids] - self.env._robot.data.root_link_pos_w[env_ids], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0
        self._lap_start_time[env_ids] = 0.0
        self._powerloop_start_time[env_ids] = 0.0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0

        self.env._crashed[env_ids] = 0

        self._visited_p1[env_ids] = False
        self._visited_p2[env_ids] = False
        self._visited_p3[env_ids] = False
        self._powerloop_done_this_lap[env_ids] = False

        # Reset 3-lap race trackers
        self._total_gates_passed[env_ids] = 0
        self._race_start_time[env_ids] = 0.0
        self._had_crash[env_ids] = False

        # Domain randomization: enabled after 5500 iterations
        if domain_randomization:
            self._randomize_parameters(env_ids, it)

        # self._last_gate_x[env_ids] = 0.0
        # self._powerloop_active[env_ids] = False

    def _randomize_parameters(self, env_ids: torch.Tensor, iteration: int):
        """Randomize dynamics parameters within evaluation bounds for the given envs."""
        n = len(env_ids)
        cfg = self.cfg

        if iteration == 1501:
            print(f"Starting domain randomization at iteration {iteration} on envs {env_ids.cpu().numpy()}")

        if (iteration > 2000):
            # TWR +-5%
            twr = cfg.thrust_to_weight * torch.empty(n, device=self.device).uniform_(0.95, 1.05)
            self.env._thrust_to_weight[env_ids] = twr
        
        if (iteration > 2500):
            # Aerodynamics: 50%-200%
            k_xy = cfg.k_aero_xy * torch.empty(n, device=self.device).uniform_(0.5, 2.0)
            k_z = cfg.k_aero_z * torch.empty(n, device=self.device).uniform_(0.5, 2.0)
            self.env._K_aero[env_ids, :2] = k_xy.unsqueeze(1)
            self.env._K_aero[env_ids, 2] = k_z
        
        if (iteration > 3000):
            # PID gains - roll/pitch: +-15%
            kp_rp = cfg.kp_omega_rp * torch.empty(n, device=self.device).uniform_(0.85, 1.15)
            ki_rp = cfg.ki_omega_rp * torch.empty(n, device=self.device).uniform_(0.85, 1.15)
            kd_rp = cfg.kd_omega_rp * torch.empty(n, device=self.device).uniform_(0.7, 1.3)
            self.env._kp_omega[env_ids, :2] = kp_rp.unsqueeze(1)
            self.env._ki_omega[env_ids, :2] = ki_rp.unsqueeze(1)
            self.env._kd_omega[env_ids, :2] = kd_rp.unsqueeze(1)

        if iteration > 3500:

            # PID gains - yaw: +-15%
            kp_y = cfg.kp_omega_y * torch.empty(n, device=self.device).uniform_(0.85, 1.15)
            ki_y = cfg.ki_omega_y * torch.empty(n, device=self.device).uniform_(0.85, 1.15)
            kd_y = cfg.kd_omega_y * torch.empty(n, device=self.device).uniform_(0.7, 1.3)
            self.env._kp_omega[env_ids, 2] = kp_y
            self.env._ki_omega[env_ids, 2] = ki_y
            self.env._kd_omega[env_ids, 2] = kd_y

        # if iteration > 7000:
        #     # Motor time constant: keep fixed (no spec given)
        #     self.env._tau_m[env_ids] = cfg.tau_m