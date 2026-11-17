"""Success rate and recovery time after stepping onto a slippery surface.

Running-only stand-in for the proposal's landing test (no jump yet). Each
trial runs forward at v_cmd ~ U(0.5, 1.5) m/s on mu = 0.8 for --switch_s
seconds, then the ground under it switches to the test friction (the
"event", where landing will go later) and the trial continues for
--after_s seconds. 200 trials per surface by default, from a fixed seed, so
every policy sees the same trials.

  success rate   trials with no fall within --after_s of the event
  recovery time  time from the event until |v_xy - v_cmd| < 0.2 m/s holds
                 for 0.5 s (trials that never recover are counted separately)
  estimator err  mean |v_est - v_true| over the 0.5 s after the event

Results are saved as JSON, so a second policy can be overlaid later:
    uv run --frozen scripts/eval_recovery.py --task go2trot
    uv run --frozen scripts/eval_recovery.py --task go2trot_truevel \
        --compare recovery_go2trot.json
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from play import get_play_args, setup  # noqa: E402

# Reference categorical palette, slots 1-2 (validated pair), plus text inks.
SERIES = ["#2a78d6", "#eb6834"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trials", type=int, default=200, help="per surface")
    parser.add_argument("--frictions", type=float, nargs="+", default=[0.8, 0.4, 0.2])
    parser.add_argument("--approach_mu", type=float, default=0.8)
    parser.add_argument("--switch_s", type=float, default=3.0)
    parser.add_argument("--after_s", type=float, default=5.0)
    parser.add_argument("--vel_tol", type=float, default=0.2)
    parser.add_argument("--hold_s", type=float, default=0.5)
    parser.add_argument("--trial_seed", type=int, default=0)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="PDF path")
    parser.add_argument("--json", type=str, default=None, help="results path")
    parser.add_argument("--compare", type=str, nargs="*", default=[])
    args, rest = parser.parse_known_args()
    play_args = get_play_args(rest)
    play_args.headless = True
    play_args.keyboard = False
    play_args.num_envs = args.trials * len(args.frictions)
    if play_args.seed is None:
        play_args.seed = args.trial_seed
    return args, play_args


def run_trials(env, runner, args):
    n_mu, n = len(args.frictions), args.trials
    surface = np.repeat(np.arange(n_mu), n)  # trial -> surface index
    rng = np.random.default_rng(args.trial_seed)
    v_cmd = rng.uniform(0.5, 1.5, size=n)  # same commands on every surface
    v_cmd = np.tile(v_cmd, n_mu)

    env.cfg.commands.resampling_time = 1e9
    env.set_ground_friction(args.approach_mu)
    env.timed_out[:] = True
    env.reset()
    env.commands[:] = 0.0
    env.commands[:, 0] = torch.as_tensor(v_cmd, dtype=torch.float, device=env.device)

    dt = env.dt
    event = int(round(args.switch_s / dt))
    steps = event + int(round(args.after_s / dt))
    true_v = np.zeros((steps, env.num_envs, 2))
    est_v = np.zeros_like(true_v)
    fell = np.zeros((steps, env.num_envs), dtype=bool)

    for k in range(steps):
        if k == event:
            for i, mu in enumerate(args.frictions):
                ids = torch.as_tensor(np.flatnonzero(surface == i), device=env.device)
                env.set_ground_friction(mu, ids)
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        est_v[k] = runner.estimated_base_lin_vel()[:, :2].cpu().numpy()
        true_v[k] = env.base_lin_vel[:, :2].cpu().numpy()
        env.step()
        fell[k] = env.terminated.cpu().numpy()
        runner.reset_envs()  # fallen trials are excluded from here on
    return surface, v_cmd, true_v, est_v, fell, event, dt


def summarize(args, surface, v_cmd, true_v, est_v, fell, event, dt):
    hold = int(round(args.hold_s / dt))
    fell_any = np.cumsum(fell, axis=0) > 0  # stays True after the first fall
    approach_fall = fell_any[event - 1]
    cmd = np.stack([v_cmd, np.zeros_like(v_cmd)], axis=1)
    err = np.linalg.norm(true_v - cmd[None], axis=2)
    est_err = np.linalg.norm(est_v - true_v, axis=2)

    results = []
    for i, mu in enumerate(args.frictions):
        trials = np.flatnonzero((surface == i) & ~approach_fall)
        success = ~fell_any[-1, trials]
        rec_times = []
        for t in trials:
            ok = (err[event:, t] < args.vel_tol) & ~fell_any[event:, t]
            run = np.convolve(ok.astype(int), np.ones(hold, dtype=int), "valid")
            hits = np.flatnonzero(run == hold)
            if hits.size:
                rec_times.append(hits[0] * dt)
        window = est_err[event : event + hold, trials]
        results.append(
            {
                "mu": mu,
                "trials": int(trials.size),
                "approach_falls": int((approach_fall & (surface == i)).sum()),
                "success_rate": float(success.mean()) if trials.size else float("nan"),
                "recovered_frac": len(rec_times) / max(trials.size, 1),
                "recovery_times": rec_times,
                "median_recovery": float(np.median(rec_times)) if rec_times else None,
                "est_err_after_event": float(window.mean()),
                "est_err_peak": float(window.max()),
            }
        )
    return results


def plot(all_runs, args, out):
    mus = [r["mu"] for r in all_runs[0]["results"]]
    x = np.arange(len(mus))
    n_runs = len(all_runs)
    width = 0.8 / n_runs
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    titles = (
        "Success rate (no fall)",
        "Recovery time after surface switch",
        "Estimator error, 0.5 s after switch",
    )
    for ax, title in zip(axes, titles):
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xticks(x, [f"$\\mu$ = {m}" for m in mus])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_2)

    for j, run in enumerate(all_runs):
        color = SERIES[j % len(SERIES)]
        offset = (j - (n_runs - 1) / 2) * width
        res = run["results"]

        rates = [100 * r["success_rate"] for r in res]
        bars = axes[0].bar(x + offset, rates, width * 0.92, color=color,
                           label=run["label"])
        for b, v in zip(bars, rates):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                         ha="center", va="bottom", fontsize=8, color=INK_2)

        for i, r in enumerate(res):
            times = np.asarray(r["recovery_times"])
            xs = np.full(times.size, x[i] + offset)
            xs = xs + np.random.default_rng(i).uniform(-width / 4, width / 4, times.size)
            axes[1].scatter(xs, times, s=10, color=color, alpha=0.35, lw=0)
            if times.size:
                axes[1].hlines(np.median(times), x[i] + offset - width / 2.5,
                               x[i] + offset + width / 2.5, color=INK, lw=2)
            axes[1].text(x[i] + offset, args.after_s * 1.02,
                         f"{100 * r['recovered_frac']:.0f}%\nrecovered",
                         ha="center", va="bottom", fontsize=8, color=INK_2)

        errs = [r["est_err_after_event"] for r in res]
        bars = axes[2].bar(x + offset, errs, width * 0.92, color=color)
        for b, v in zip(bars, errs):
            axes[2].text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                         ha="center", va="bottom", fontsize=8, color=INK_2)

    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("trials [%]", color=INK_2)
    axes[1].set_ylim(0, args.after_s * 1.18)
    axes[1].set_ylabel("seconds (bar = median)", color=INK_2)
    axes[2].axhline(0.2, color=INK_2, ls="--", lw=1)
    axes[2].text(x[-1] + 0.45, 0.2, "0.2 m/s\ntarget", va="center", ha="left",
                 fontsize=8, color=INK_2)
    axes[2].set_ylabel("mean |v_est - v_true| [m/s]", color=INK_2)
    if n_runs > 1:
        axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(
        f"Running at v_cmd ~ U(0.5, 1.5) m/s, switched from $\\mu$ = "
        f"{args.approach_mu} at t = {args.switch_s:g} s "
        f"({all_runs[0]['results'][0]['trials']} trials per surface)"
        + ("" if n_runs > 1 else f": {all_runs[0]['label']}"),
        color=INK, fontsize=11, x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(out)


def main():
    args, play_args = get_args()
    label = args.label or play_args.task
    out = args.out or f"recovery_{play_args.task}.pdf"
    json_path = args.json or f"recovery_{play_args.task}.json"
    with torch.no_grad():
        env, runner = setup(play_args)
        raw = run_trials(env, runner, args)
    results = summarize(args, *raw)
    run = {"label": label, "results": results}
    with open(json_path, "w") as f:
        json.dump(run, f, indent=1)

    for r in results:
        med = r["median_recovery"]
        print(
            f"mu={r['mu']:.1f}: success {100 * r['success_rate']:5.1f}% | "
            f"recovered {100 * r['recovered_frac']:5.1f}% | "
            f"median recovery {'-' if med is None else f'{med:.2f} s'} | "
            f"est err {r['est_err_after_event']:.3f} m/s "
            f"(peak {r['est_err_peak']:.3f}) | approach falls {r['approach_falls']}"
        )
    all_runs = [run] + [json.load(open(p)) for p in args.compare]
    plot(all_runs, args, out)
    print(f"saved {out} and {json_path}")


if __name__ == "__main__":
    main()
