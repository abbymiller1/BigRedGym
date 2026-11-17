import os

import torch
from tensordict import TensorDict

from learning.algorithms import StateEstimator, get_algorithm_class
from learning.modules import Actor, Critic, StateEstimatorNN
from learning.storage import DictStorage
from learning.utils import Logger

from .on_policy_runner import OnPolicyRunner

logger = Logger()
storage = DictStorage()


class EstimatorRunner(OnPolicyRunner):
    """PPO plus a concurrently trained state estimator (Ji et al. 2022).

    actor input = [noisy o_t, x_hat], with x_hat = estimator(noisy o_t).
    use_true_velocity=True swaps x_hat's velocity for the true one, while
    the estimator still trains so its error can be compared.
    """

    def parse_train_cfg(self, train_cfg):
        super().parse_train_cfg(train_cfg)
        self.se_cfg = train_cfg["state_estimator"]

    def _set_up_alg(self):
        num_raw_obs = self.get_obs_size(self.actor_cfg["obs"])
        num_se = self.get_obs_size(self.se_cfg["targets"])
        num_actions = self.get_action_size(self.actor_cfg["actions"])
        num_critic_obs = self.get_obs_size(self.critic_cfg["obs"])

        actor = Actor(num_raw_obs + num_se, num_actions, **self.actor_cfg)
        critic = Critic(num_critic_obs, **self.critic_cfg)
        alg_class = get_algorithm_class(self.cfg["algorithm_class_name"])
        self.alg = alg_class(actor, critic, device=self.device, **self.alg_cfg)

        se_net = StateEstimatorNN(num_raw_obs, num_se, **self.se_cfg["network"])
        self.se = StateEstimator(
            se_net, device=self.device, **self.se_cfg["algorithm"]
        )
        self.se.mean_loss = 0.0
        self.se.vel_rmse = 0.0
        self._noise_cache = {}

    # ---------------- observations ----------------
    def get_noise(self, obs_list, noise_dict):
        """Uniform U(-a, a) noise as in the paper (base runner is Gaussian)."""
        key = tuple(obs_list)
        if key not in self._noise_cache:
            half_width = torch.zeros(self.get_obs_size(obs_list), device=self.device)
            idx = 0
            for obs in obs_list:
                size = self.get_obs_size([obs])
                if obs in noise_dict:
                    width = torch.ones(size, device=self.device) * torch.tensor(
                        noise_dict[obs], device=self.device
                    )
                    if obs in self.env.scales:
                        width = width / self.env.scales[obs]
                    half_width[idx : idx + size] = width
                idx += size
            self._noise_cache[key] = half_width
        half_width = self._noise_cache[key]
        u = 2.0 * torch.rand(self.env.num_envs, half_width.numel(), device=self.device)
        return half_width * (u - 1.0)

    @torch.no_grad()
    def get_actor_obs(self, noisy=True):
        obs_list = self.actor_cfg["obs"]
        if noisy:
            raw_obs = self.get_noisy_obs(obs_list, self.actor_cfg["noise"])
        else:
            raw_obs = self.get_obs(obs_list)
        se_targets = self.get_obs(self.se_cfg["targets"])
        estimate = self.se.predict(raw_obs)
        if self.se_cfg["use_true_velocity"]:
            policy_state = torch.cat((se_targets[:, :3], estimate[:, 3:]), dim=1)
        else:
            policy_state = estimate
        actor_obs = torch.cat((raw_obs, policy_state), dim=1)
        self.last_estimate = estimate
        return actor_obs, raw_obs, se_targets, estimate

    def estimated_base_lin_vel(self):
        """Latest estimated body-frame velocity [m/s], shape (num_envs, 3)."""
        scale = torch.as_tensor(self.env.scales["base_lin_vel"], device=self.device)
        return self.last_estimate[:, :3] * scale

    # ---------------- training ----------------
    def learn(self, states_to_log_dict=None):
        n_policy_steps = int((1 / self.env.dt) / self.actor_cfg["frequency"])
        assert n_policy_steps > 0, "actor frequency should be less than ctrl_freq"
        self.set_up_logger(dt=self.env.dt * n_policy_steps)
        rewards_dict = self.initialize_rewards_dict(n_policy_steps)
        vel_scale = torch.as_tensor(
            self.env.scales["base_lin_vel"], device=self.device
        )
        ji_names = getattr(self.env, "ji_term_names", ())

        self.alg.switch_to_train()
        self.se.state_estimator.train()
        actor_obs, raw_obs, se_targets, estimate = self.get_actor_obs(noisy=False)
        critic_obs = self.get_obs(self.critic_cfg["obs"])
        tot_iter = self.it + self.num_learning_iterations
        self.save()

        transition = TensorDict({}, batch_size=self.env.num_envs, device=self.device)
        transition.update(
            {
                "actor_obs": actor_obs,
                "next_actor_obs": actor_obs,
                "actions": self.alg.act(actor_obs),
                "critic_obs": critic_obs,
                "next_critic_obs": critic_obs,
                "rewards": self.get_rewards({"termination": 0.0})["termination"],
                "timed_out": self.get_timed_out(),
                "terminated": self.get_terminated(),
                "dones": self.get_timed_out() | self.get_terminated(),
            }
        )
        storage.initialize(
            transition,
            self.env.num_envs,
            self.env.num_envs * self.num_steps_per_env,
            device=self.device,
        )
        self.se.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [raw_obs.shape[1]],
            [se_targets.shape[1]],
        )

        if self.actor_cfg["normalize_obs"] or self.critic_cfg["normalize_obs"]:
            self.burn_in_normalization()
            actor_obs, raw_obs, se_targets, estimate = self.get_actor_obs()
            critic_obs = self.get_obs(self.critic_cfg["obs"])

        logger.tic("runtime")
        for self.it in range(self.it + 1, tot_iter + 1):
            self.env.training_iteration = self.it  # command curriculum
            logger.tic("iteration")
            logger.tic("collection")
            vel_sq_sum = 0.0

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs)
                    self.set_actions(
                        self.actor_cfg["actions"],
                        actions,
                        self.actor_cfg["disable_actions"],
                    )
                    transition.update(
                        {
                            "actor_obs": actor_obs,
                            "actions": actions,
                            "critic_obs": critic_obs,
                        }
                    )
                    # Supervised sample: (noisy o_t, true state at t)
                    self.se.process_env_step(raw_obs, se_targets)
                    vel_err = (estimate[:, :3] - se_targets[:, :3]) * vel_scale
                    vel_sq_sum = vel_sq_sum + vel_err.square().sum(dim=1).mean()

                    for step in range(n_policy_steps):
                        self.env.step()
                        self.update_rewards_dict(rewards_dict, step)
                    if ji_names and self.env.ji_terms:
                        logger.log_rewards(
                            {
                                f"ji_{k}": v * self.env.dt
                                for k, v in self.env.ji_terms.items()
                            }
                        )

                    self.reset_envs()

                    total_rewards = torch.stack(
                        tuple(rewards_dict.sum(dim=0).values())
                    ).sum(dim=0)
                    actor_obs, raw_obs, se_targets, estimate = self.get_actor_obs()
                    critic_obs = self.get_obs(self.critic_cfg["obs"])

                    transition.update(
                        {
                            "next_actor_obs": actor_obs,
                            "next_critic_obs": critic_obs,
                            "rewards": total_rewards,
                            "timed_out": self.env.timed_out,
                            "dones": self.env.timed_out | self.env.terminated,
                        }
                    )
                    storage.add_transitions(transition)

                    logger.log_rewards(rewards_dict.sum(dim=0))
                    logger.log_rewards({"total_rewards": total_rewards})
                    logger.finish_step(self.env.timed_out | self.env.terminated)
            logger.toc("collection")

            logger.tic("learning")
            self.alg.update(storage.data)
            storage.clear()
            self.se.mean_loss = self.se.update()
            self.se.vel_rmse = float(torch.sqrt(vel_sq_sum / self.num_steps_per_env))
            logger.toc("learning")
            logger.log_all_categories()

            logger.finish_iteration()
            logger.toc("iteration")
            logger.toc("runtime")
            logger.print_to_terminal()
            print(
                f"{'state estimator loss':>35}: {self.se.mean_loss:.4f}\n"
                f"{'velocity estimate RMSE':>35}: {self.se.vel_rmse:.3f} m/s\n"
            )

            if self.it % self.save_interval == 0:
                self.save()
        self.save()

    @torch.no_grad()
    def burn_in_normalization(self, n_iterations=100):
        actor_obs, *_ = self.get_actor_obs()
        for _ in range(n_iterations):
            actions = self.alg.act(actor_obs)
            self.set_actions(self.actor_cfg["actions"], actions)
            self.env.step()
            actor_obs, *_ = self.get_actor_obs()
            self.alg.critic.evaluate(self.get_obs(self.critic_cfg["obs"]))
        self.env.reset()

    def set_up_logger(self, dt=None):
        if dt is None:
            dt = self.env.dt
        logger.initialize(
            self.env.num_envs,
            dt,
            self.cfg["max_iterations"],
            self.device,
            log_dir=self.log_dir,
        )
        logger.register_rewards(list(self.critic_cfg["reward"]["weights"].keys()))
        logger.register_rewards(
            list(self.critic_cfg["reward"]["termination_weight"].keys())
        )
        logger.register_rewards(["total_rewards"])
        names = getattr(self.env, "ji_term_names", ())
        if names:
            logger.register_rewards([f"ji_{n}" for n in names])
        logger.register_category(
            "algorithm",
            self.alg,
            ["mean_value_loss", "mean_surrogate_loss", "learning_rate"],
        )
        logger.register_category("actor", self.alg.actor, ["action_std", "entropy"])
        logger.register_category("state_estimator", self.se, ["mean_loss", "vel_rmse"])
        logger.attach_torch_obj_to_wandb((self.alg.actor, self.alg.critic))

    # ---------------- save / load / play ----------------
    def _estimator_path(self, model_path):
        folder, name = os.path.split(model_path)
        return os.path.join(folder, name.replace("model_", "estimator_"))

    def save(self):
        super().save()
        path = self._estimator_path(
            os.path.join(self.log_dir, "model_{}.pt".format(self.it))
        )
        torch.save(
            {
                "state_estimator_state_dict": self.se.state_estimator.state_dict(),
                "optimizer_state_dict": self.se.optimizer.state_dict(),
                "iter": self.it,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        super().load(path, load_optimizer)
        loaded = torch.load(
            self._estimator_path(path), weights_only=True, map_location=self.device
        )
        self.se.state_estimator.load_state_dict(loaded["state_estimator_state_dict"])
        if load_optimizer:
            self.se.optimizer.load_state_dict(loaded["optimizer_state_dict"])

    def switch_to_eval(self):
        super().switch_to_eval()
        self.se.state_estimator.eval()

    def get_inference_actions(self):
        actor_obs, *_ = self.get_actor_obs(noisy=False)
        return self.alg.actor.act_inference(actor_obs)

    def export(self, path):
        super().export(path)
        self.se.export(path)
