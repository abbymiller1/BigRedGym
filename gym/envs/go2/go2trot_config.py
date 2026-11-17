import math

from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2TrotCfg(Go2Cfg):
    """Go2 running, following Ji et al. (RA-L 2022). Gait is not prescribed."""

    class env(Go2Cfg.env):
        num_envs = 2**12
        episode_length_s = 10
        history_steps = [2, 4, 6]  # joint history at t-0.02/0.04/0.06 s

    class init_state(Go2Cfg.init_state):
        # q_nominal. Key names match go2_deploy/deploy_config.py, which reads them.
        default_joint_angles = {
            "hip_joint": 0.0,
            "calf_joint": -1.36,
            "FL_thigh_joint": 0.96,
            "FR_thigh_joint": 0.96,
            "RL_thigh_joint": 0.96,
            "RR_thigh_joint": 0.96,
        }
        reset_mode = "reset_to_range"
        # Paper Table I: joints +/-0.2 rad, +/-2.5 rad/s around nominal
        dof_pos_range = {"hip": [-0.2, 0.2], "thigh": [0.76, 1.16], "calf": [-1.56, -1.16]}
        dof_vel_range = {"hip": [-2.5, 2.5], "thigh": [-2.5, 2.5], "calf": [-2.5, 2.5]}
        root_pos_range = [
            [0.0, 0.0], [0.0, 0.0], [0.40, 0.45],          # x, y, z
            [-0.2, 0.2], [-0.2, 0.2], [-math.pi, math.pi],  # roll, pitch, yaw
        ]
        root_vel_range = [
            [-1.0, 1.0], [-0.5, 0.5], [-0.5, 0.5],          # vx, vy, vz
            [-0.7, 0.7], [-0.7, 0.7], [-0.7, 0.7],          # wx, wy, wz
        ]

    class control(Go2Cfg.control):
        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 20.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 0.5}
        ctrl_frequency = 100  # paper: policy + estimator at 100 Hz
        desired_sim_frequency = 200
        motor_friction_range = {"hip": [0.0, 0.3], "thigh": [0.0, 0.3], "calf": [0.1, 0.7]}

    class commands(Go2Cfg.commands):
        resampling_time = 5.0
        zero_command_prob = 0.1
        curriculum = True

        class ranges(Go2Cfg.commands.ranges):
            lin_vel_x_start = [-0.5, 1.0]
            lin_vel_x = [-1.75, 3.5]  # paper; [-1.0, 2.0] trains faster for your tests
            lin_vel_y = 1.0
            yaw_vel = 2.0

    class push_robots(Go2Cfg.push_robots):
        toggle = False  # the paper does not push

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            contact_friction_range = [0.4, 1.0]
            link_mass_scale_range = None  # paper does not randomize mass

        class episode(Go2Cfg.domain_randomization.episode):
            # Paper: Kp +/-2, Kd +/-0.1 around 20 / 0.5
            scale_ranges = {"p_gains": [0.9, 1.1], "d_gains": [0.8, 1.2]}

    class asset(Go2Cfg.asset):
        # Paper: any contact other than knees/feet terminates (-10)
        terminate_after_contacts_on = ["base", "Head", "hip", "thigh"]
        penalize_contacts_on = []
        foot_radius = 0.022  # go2.urdf

    class reward_settings(Go2Cfg.reward_settings):
        foot_clearance_target = 0.09  # [m]
        contact_force_threshold = 1.0  # [N]

        class ji:  # Ji et al. Table II
            lin_vel = 3.0
            ang_vel = 3.0
            air_time = 0.3
            foot_slip = -0.08
            foot_clearance = -15.0
            orientation = -3.0
            torque = -6e-4
            joint_pos = -0.75
            joint_vel = -6e-4
            joint_acc = -0.02
            smooth1 = -2.5
            smooth2 = -1.2
            base_motion = -1.5
            neg_scale = 0.2

    class scaling(Go2Cfg.scaling):
        dof_pos_target = 4 * [0.2, 0.3, 0.3]  # Go2 needs more than the paper's 0.1
        dof_pos_target_hist = 2 * Go2Cfg.scaling.dof_pos  # list repeat -> 24
        dof_vel_hist = 3 * Go2Cfg.scaling.dof_vel  # -> 36


class Go2TrotRunnerCfg(Go2RunnerCfg):
    runner_class_name = "EstimatorRunner"

    class actor(Go2RunnerCfg.actor):
        hidden_dims = [512, 256, 64]
        activation = "elu"
        normalize_obs = True
        obs = [
            "projected_gravity", "base_ang_vel", "dof_pos_obs", "dof_vel",
            "dof_pos_target_hist", "dof_pos_err_hist", "dof_vel_hist",
            "foot_pos_body", "commands",
        ]
        actions = ["dof_pos_target"]

        class noise:  # uniform half-widths U(-a, a), physical units (paper)
            projected_gravity = 0.03
            base_ang_vel = 0.1
            dof_pos_obs = 0.05
            dof_vel = 0.5
            dof_pos_err_hist = 0.05
            dof_vel_hist = 0.5
            foot_pos_body = 4 * [0.01, 0.005, 0.02]

    class critic(Go2RunnerCfg.critic):
        hidden_dims = [512, 256, 64]
        activation = "elu"
        normalize_obs = True
        obs = [
            "base_lin_vel", "base_height", "foot_height", "foot_contact",
            "projected_gravity", "base_ang_vel", "dof_pos_obs", "dof_vel",
            "dof_pos_target_hist", "foot_pos_body", "commands",
        ]

        class reward:
            class weights:
                ji_total = 1.0

            class termination_weight:
                termination = 10.0  # rewards are all scaled by dt -> paper's -10

    class state_estimator:
        targets = ["base_lin_vel", "foot_height", "foot_contact"]  # velocity first
        use_true_velocity = False

        class network:
            hidden_dims = [256, 128]
            activation = "elu"
            dropouts = 0.0

        class algorithm:
            learning_rate = 1.0e-3
            num_mini_batches = 8
            num_learning_epochs = 4

    class algorithm(Go2RunnerCfg.algorithm):
        entropy_coef = 0.001

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2trot"
        max_iterations = 2500


class Go2TrotTrueVelRunnerCfg(Go2TrotRunnerCfg):
    """Baseline: identical, but the actor gets the TRUE body velocity."""

    class state_estimator(Go2TrotRunnerCfg.state_estimator):
        use_true_velocity = True

    class runner(Go2TrotRunnerCfg.runner):
        experiment_name = "go2trot_truevel"