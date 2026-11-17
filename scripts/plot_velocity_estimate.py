"""Plot estimated vs. true base velocity on normal and slippery ground.

Usage:
    uv run --frozen scripts/plot_velocity_estimate.py --task=go2trot
    uv run --frozen scripts/plot_velocity_estimate.py --task=go2trot \
        --load_run <run_dir> --forward_vel 1.0 --frictions 0.8 0.6 0.4 0.2
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from play import get_play_args, setup  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", type=str, default="velocity_estimate.pdf")
    parser.add_argument("--duration_s", type=float, default=8.0)
    parser.add_argument("--forward_vel", type=float, default=1.0)
    parser.add_argument("--frictions", type=float, nargs="+", default=[0.8, 0.6, 0.4, 0.2])
    args, rest = parser.parse_known_args()
    play_args = get_play_args(rest)
    play_args.headless = True
    play_args.keyboard = False
    return args, play_args


def run_surface(env, runner, mu, duration_s, forward_vel):
    env.set_ground_friction(mu)
    env.timed_out[:] = True
    env.reset()
    env.commands[:] = 0.0
    env.commands[:, 0] = forward_vel

    steps = int(round(duration_s / env.dt))
    times, true_v, est_v = [], [], []
    for i in range(steps):
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        est_v.append(runner.estimated_base_lin_vel()[:, :2].cpu().numpy().copy())
        true_v.append(env.base_lin_vel[:, :2].cpu().numpy().copy())
        times.append(i * env.dt)
        env.step()
        runner.reset_envs()
    return np.array(times), np.stack(true_v), np.stack(est_v)


def main():
    args, play_args = get_args()
    with torch.no_grad():
        env, runner = setup(play_args)
        env.cfg.commands.resampling_time = 1e9  # hold the command

        results = []
        for mu in args.frictions:
            t, true_v, est_v = run_surface(env, runner, mu, args.duration_s, args.forward_vel)
            settle = t >= 1.0  # skip start-up transient
            err = np.linalg.norm(est_v - true_v, axis=2)[settle]
            rms = float(np.sqrt(np.mean(err**2)))
            peak = float(err.max())
            results.append((mu, t, true_v, est_v, rms, peak))
            print(f"mu={mu:.2f}: RMS error {rms:.3f} m/s, peak {peak:.3f} m/s")

    fig, axes = plt.subplots(len(results), 1, figsize=(9, 2.6 * len(results)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (mu, t, true_v, est_v, rms, peak) in zip(axes, results):
        ax.plot(t, true_v[:, 0, 0], color="k", lw=1.5, label="true $v_x$")
        ax.plot(t, est_v[:, 0, 0], color="C1", lw=1.2, label="estimated $v_x$")
        ax.axhline(args.forward_vel, color="0.6", ls="--", lw=1, label="command")
        ax.set_ylabel("$v_x$ [m/s]")
        ax.set_title(f"$\\mu$ = {mu}: RMS err {rms:.3f} m/s, peak {peak:.3f} m/s (all envs)")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()