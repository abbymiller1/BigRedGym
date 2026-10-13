"""Plot how well a trained go2trot policy tracks its base-height command.

Runs the policy headless through a scripted mid -> low -> high -> low -> high
schedule and saves a time-series plot of the commanded vs. measured base
height (mean +/- std across environments, plus environment 0) as a PDF.

Usage:
    uv run --frozen scripts/plot_height_tracking.py --task=go2trot
    uv run --frozen scripts/plot_height_tracking.py --task=go2trot \
        --load_run <run_dir> --forward_vel 0.5 --out height_tracking.pdf
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.play import get_play_args, setup  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", type=str, default="height_tracking.pdf")
    parser.add_argument(
        "--hold_s", type=float, default=4.0, help="seconds per command segment"
    )
    parser.add_argument(
        "--forward_vel", type=float, default=0.0, help="forward velocity command"
    )
    args, rest = parser.parse_known_args()
    play_args = get_play_args(rest)
    play_args.headless = True
    return args, play_args


def run_schedule(env, runner, hold_s, forward_vel):
    # Offsets from the middle of the trained range: mid, low, high, low, high.
    lo, hi = -env.height_command_max, env.height_command_max
    schedule = [0.0, lo, hi, lo, hi]
    steps_per_segment = int(round(hold_s / env.dt))

    env.commands[:] = 0.0
    env.commands[:, 0] = forward_vel
    times, commanded, measured = [], [], []
    for k, offset in enumerate(schedule):
        env.height_command[:] = offset
        for i in range(steps_per_segment):
            runner.set_actions(
                runner.actor_cfg["actions"],
                runner.get_inference_actions(),
                runner.actor_cfg["disable_actions"],
            )
            env.step()
            env.check_exit()
            times.append((k * steps_per_segment + i + 1) * env.dt)
            commanded.append(env.base_height_target()[:, 0].cpu().numpy().copy())
            measured.append(env.base_height[:, 0].cpu().numpy().copy())
    return (
        np.array(times),
        np.stack(commanded),
        np.stack(measured),
        steps_per_segment,
        schedule,
    )


def plot(times, commanded, measured, steps_per_segment, schedule, out):
    mean, std = measured.mean(axis=1), measured.std(axis=1)
    error = measured - commanded
    rmse = np.sqrt(np.mean(error**2))

    fig, (ax_h, ax_e) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_h.step(times, commanded[:, 0], where="post", color="k", lw=1.5,
              label="commanded")
    ax_h.plot(times, mean, color="C0", lw=1.5,
              label=f"measured, mean of {measured.shape[1]} envs")
    ax_h.fill_between(times, mean - std, mean + std, color="C0", alpha=0.25,
                      label="measured, $\\pm$1 std")
    ax_h.plot(times, measured[:, 0], color="C1", lw=0.8, alpha=0.8,
              label="measured, env 0")
    ax_h.set_ylabel("base height [m]")
    ax_h.set_title(f"Go2 base-height tracking (overall RMSE {100 * rmse:.1f} cm)")
    ax_h.legend(loc="lower right", fontsize=8)
    ax_h.grid(alpha=0.3)

    ax_e.plot(times, error.mean(axis=1), color="C3", lw=1.2)
    ax_e.axhline(0.0, color="k", lw=0.8)
    ax_e.set_ylabel("error [m]")
    ax_e.set_xlabel("time [s]")
    ax_e.grid(alpha=0.3)

    # Steady-state error per segment: last half of each hold.
    print(f"overall RMSE: {100 * rmse:.2f} cm")
    for k, _ in enumerate(schedule):
        start = k * steps_per_segment + steps_per_segment // 2
        end = (k + 1) * steps_per_segment
        seg = error[start:end]
        target = commanded[start, 0]
        print(
            f"segment {k} target {target:.3f} m: steady-state mean error "
            f"{100 * seg.mean():+.2f} cm, RMSE {100 * np.sqrt(np.mean(seg**2)):.2f} cm"
        )
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    args, play_args = get_args()
    with torch.no_grad():
        env, runner = setup(play_args)
        if not hasattr(env, "height_command"):
            raise SystemExit(f"task {play_args.task!r} has no height_command")
        plot(*run_schedule(env, runner, args.hold_s, args.forward_vel), args.out)