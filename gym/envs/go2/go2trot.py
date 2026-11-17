import math

import torch

from gym.envs.base.legged_robot import LeggedRobot
from gym.utils.sampling import masked_update
from gym.utils.torch_quat import quat_rotate_inverse

# Ji et al. (RA-L 2022), Table II. r_tot = r_pos * exp(0.2 * r_neg)
POSITIVE_TERMS = ("lin_vel", "ang_vel", "air_time")
JI_TERM_NAMES = (
    "lin_vel", "ang_vel", "air_time",
    "foot_slip", "foot_clearance", "orientation", "torque", "joint_pos",
    "joint_vel", "joint_acc", "smooth1", "smooth2", "base_motion",
)


class Go2Trot(LeggedRobot):
    """Go2 running with a concurrent state estimator (Ji et al. 2022).

    No gait phase, oscillator or reference trajectory: the gait emerges
    from the airtime, clearance and smoothness rewards.
    """

    ji_term_names = JI_TERM_NAMES

    def __init__(self, cfg, device, headless, backend):
        # Set before super().__init__, because reset() resamples commands.
        self.training_iteration = 0
        super().__init__(cfg, device, headless, backend)

    def _zeros(self, *shape):
        return torch.zeros(self.num_envs, *shape, dtype=torch.float, device=self.device)

    # ---------------- buffers ----------------
    def _init_buffers(self):
        super()._init_buffers()
        n = self.num_actuators
        nf = len(self.feet_indices)

        # Estimator targets (true simulator values)
        self.foot_height = self._zeros(nf)
        self.foot_contact = self._zeros(nf)

        # Extra policy observations (paper Eq. 3)
        self.foot_pos_body = self._zeros(3 * nf)
        self.dof_pos_target_hist = self._zeros(2 * n)  # q_des at t-1, t-2
        steps = self.cfg.env.history_steps  # [2, 4, 6] control steps back
        self._hist_steps = torch.tensor(steps, dtype=torch.long, device=self.device)
        buf_len = max(steps) + 1
        self._dof_err_buf = self._zeros(buf_len, n)
        self._dof_vel_buf = self._zeros(buf_len, n)
        self.dof_pos_err_hist = self._zeros(len(steps) * n)
        self.dof_vel_hist = self._zeros(len(steps) * n)

        # Reward bookkeeping
        self.last_dof_vel = self._zeros(n)
        self.feet_air_time = self._zeros(nf)
        self.feet_stance_time = self._zeros(nf)
        self.foot_vel_xy = self._zeros(nf, 2)
        self.ji_terms = {}

        # Motor Coulomb friction, resampled every episode
        ranges = self.cfg.control.motor_friction_range
        low, high = [], []
        for name in self.actuated_dof_names:
            key = [k for k in ranges if k in name][0]
            low.append(ranges[key][0])
            high.append(ranges[key][1])
        self._motor_friction_low = torch.tensor(low, device=self.device)
        self._motor_friction_high = torch.tensor(high, device=self.device)
        self.motor_friction = self._zeros(n)

        self._update_foot_states()

    # ---------------- step hooks ----------------
    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self.last_dof_vel[:] = self.dof_vel.index_select(1, self.actuated_dof_indices)

    def _post_decimation_step(self):
        super()._post_decimation_step()  # rolls dof_pos_history, resamples cmds
        self._update_foot_states()
        self._update_air_time()
        self._update_joint_history()

    def _compute_torques(self):
        torques = super()._compute_torques()
        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        torques = torques - self.motor_friction * torch.tanh(vel / 0.05)
        return torch.clip(torques, -self.actuated_torque_limits, self.actuated_torque_limits)

    def _update_foot_states(self):
        nf = len(self.feet_indices)
        foot_pos = self._rigid_body_pos[:, self.feet_indices, :]
        foot_vel = self._rigid_body_lin_vel[:, self.feet_indices, :]
        self.foot_height[:] = foot_pos[..., 2] - self.cfg.asset.foot_radius
        self.foot_vel_xy[:] = foot_vel[..., :2]
        force_z = self.contact_forces[:, self.feet_indices, 2]
        self.foot_contact[:] = (
            force_z > self.cfg.reward_settings.contact_force_threshold
        ).float()
        rel = foot_pos - self.root_states[:, :3].unsqueeze(1)
        quat = self.base_quat.unsqueeze(1).expand(-1, nf, -1).reshape(-1, 4)
        self.foot_pos_body[:] = quat_rotate_inverse(
            quat, rel.reshape(-1, 3)
        ).reshape(self.num_envs, -1)

    def _update_air_time(self):
        contact = self.foot_contact > 0.5
        self.feet_air_time = torch.where(contact, 0.0, self.feet_air_time + self.dt)
        self.feet_stance_time = torch.where(contact, self.feet_stance_time + self.dt, 0.0)

    def _update_joint_history(self):
        n = self.num_actuators
        self.dof_pos_target_hist[:] = self.dof_pos_history[:, : 2 * n]
        pos = self.dof_pos.index_select(1, self.actuated_dof_indices)
        default = self.default_dof_pos.index_select(1, self.actuated_dof_indices)
        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        self._dof_err_buf[:] = torch.roll(self._dof_err_buf, 1, dims=1)
        self._dof_err_buf[:, 0] = self.dof_pos_target + default - pos
        self._dof_vel_buf[:] = torch.roll(self._dof_vel_buf, 1, dims=1)
        self._dof_vel_buf[:, 0] = vel
        self.dof_pos_err_hist[:] = self._dof_err_buf[:, self._hist_steps].flatten(1)
        self.dof_vel_hist[:] = self._dof_vel_buf[:, self._hist_steps].flatten(1)

    # ---------------- resets, commands, friction ----------------
    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        n = self.num_actuators
        # Base class sets the residual target to default_dof_pos; it is an
        # offset from default, so zero it.
        self.dof_pos_target.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        masked_update(self.dof_pos_history, self.dof_pos_target.tile(1, 3), reset_mask)
        masked_update(self.dof_pos_target_hist, self.dof_pos_target.tile(1, 2), reset_mask)

        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        masked_update(self.last_dof_vel, vel, reset_mask)
        masked_update(self._dof_err_buf, torch.zeros_like(self._dof_err_buf), reset_mask)
        masked_update(
            self._dof_vel_buf, vel.unsqueeze(1).expand_as(self._dof_vel_buf), reset_mask
        )
        masked_update(
            self.dof_pos_err_hist, torch.zeros_like(self.dof_pos_err_hist), reset_mask
        )
        masked_update(self.dof_vel_hist, vel.tile(1, len(self._hist_steps)), reset_mask)
        self.feet_air_time.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self.feet_stance_time.masked_fill_(reset_mask.unsqueeze(1), 0.0)

        friction = self._motor_friction_low + (
            self._motor_friction_high - self._motor_friction_low
        ) * torch.rand(self.num_envs, n, device=self.device)
        masked_update(self.motor_friction, friction, reset_mask)
        self._update_foot_states()

    def _resample_commands(self, command_mask):
        if self.cfg.commands.curriculum:
            self.command_ranges["lin_vel_x"] = self._lin_vel_x_curriculum()
        super()._resample_commands(command_mask)
        # 10% zero commands so the policy learns to stand still
        zero = command_mask & (
            torch.rand(self.num_envs, device=self.device)
            < self.cfg.commands.zero_command_prob
        )
        self.commands.masked_fill_(zero.unsqueeze(1), 0.0)

    def set_ground_friction(self, mu, env_ids=None):
        """Set foot-ground sliding friction (0.2 = slippery). Used by teleop/eval."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        values = torch.full((len(env_ids),), float(mu), device=self.device)
        self._backend.set_contact_friction(env_ids, values)

    def _lin_vel_x_curriculum(self):
        """Paper: v_x,max = 1 + 2.5 / (1 + exp(-0.002 (t - 1000))),
        renormalized so the range equals lin_vel_x_start at t = 0."""
        start = self.cfg.commands.ranges.lin_vel_x_start
        final = self.cfg.commands.ranges.lin_vel_x
        s0 = 1.0 / (1.0 + math.exp(2.0))
        s = 1.0 / (1.0 + math.exp(-0.002 * (self.training_iteration - 1000)))
        frac = (s - s0) / (1.0 - s0)
        return [a + frac * (b - a) for a, b in zip(start, final)]

    # ---------------- reward (Ji et al. Table II) ----------------
    def _ji_reward_terms(self):
        k = self.cfg.reward_settings.ji
        n = self.num_actuators
        cmd = self.commands
        dof_vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        terms = {}

        lin_err = torch.sum(torch.square(cmd[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        terms["lin_vel"] = k.lin_vel * torch.exp(-lin_err)
        terms["ang_vel"] = k.ang_vel * torch.exp(
            -1.5 * torch.square(cmd[:, 2] - self.base_ang_vel[:, 2])
        )

        # Airtime: reward phases up to 0.2 s, zero once a phase passes 0.25 s.
        t_a, t_s = self.feet_air_time, self.feet_stance_time
        standing = (torch.norm(cmd, dim=1) < 1e-3).unsqueeze(1)
        phase_time = torch.maximum(t_a, t_s)
        r_stand = torch.clip(t_s - t_a, -0.3, 0.3)
        r_move = torch.where(phase_time < 0.25, torch.clamp(phase_time, max=0.2), 0.0)
        terms["air_time"] = k.air_time * torch.where(standing, r_stand, r_move).sum(dim=1)

        foot_speed = torch.norm(self.foot_vel_xy, dim=2)
        terms["foot_slip"] = k.foot_slip * torch.sum(
            self.foot_contact * foot_speed.square(), dim=1
        )
        terms["foot_clearance"] = k.foot_clearance * torch.sum(
            torch.square(self.foot_height - self.cfg.reward_settings.foot_clearance_target)
            * torch.sqrt(foot_speed),
            dim=1,
        )
        tilt = torch.acos(torch.clamp(-self.projected_gravity[:, 2], -1.0, 1.0))
        terms["orientation"] = k.orientation * tilt.square()
        terms["torque"] = k.torque * torch.sum(self.torques.square(), dim=1)
        terms["joint_pos"] = k.joint_pos * torch.sum(
            torch.square(self.dof_pos - self.default_dof_pos), dim=1
        )
        terms["joint_vel"] = k.joint_vel * torch.sum(dof_vel.square(), dim=1)
        terms["joint_acc"] = k.joint_acc * torch.sum(
            torch.square(dof_vel - self.last_dof_vel), dim=1
        )
        q0 = self.dof_pos_history[:, :n]
        q1 = self.dof_pos_history[:, n : 2 * n]
        q2 = self.dof_pos_history[:, 2 * n :]
        terms["smooth1"] = k.smooth1 * torch.sum(torch.square(q0 - q1), dim=1)
        terms["smooth2"] = k.smooth2 * torch.sum(torch.square(q0 - 2 * q1 + q2), dim=1)
        terms["base_motion"] = k.base_motion * (
            0.8 * self.base_lin_vel[:, 2].square()
            + 0.2 * self.base_ang_vel[:, 0].abs()
            + 0.2 * self.base_ang_vel[:, 1].abs()
        )
        return terms

    def _reward_ji_total(self):
        terms = self._ji_reward_terms()
        r_pos = sum(terms[name] for name in POSITIVE_TERMS)
        r_neg = sum(v for name, v in terms.items() if name not in POSITIVE_TERMS)
        self.ji_terms = terms  # logged by EstimatorRunner
        return r_pos * torch.exp(self.cfg.reward_settings.ji.neg_scale * r_neg)