"""Compare planners over N episodes each, on paired seeds, with artifacts.

Every planner replays the same seed list, so each faces the identical world
(spawn, enemies, food) episode-for-episode; differences in survival are then
attributable to the planner, not the draw.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from slither_bot.core import Backend
from slither_bot.eval.runner import (
    EpisodeResult,
    make_backend,
    make_viz,
    parse_params,
    run_episode,
)
from slither_bot.planners import PLANNER_NAMES, make_planner


def split_params(params: dict[str, str], planner_names: list[str]) -> dict[str, dict[str, str]]:
    """Split ``{"cbf.gamma": "2", "k_max": "30"}`` into per-planner dicts.

    Prefixed keys (``cbf.gamma=2``) go to that planner only; unprefixed keys
    go to every planner, which is only allowed when one planner is being run.
    """
    per_planner: dict[str, dict[str, str]] = {name: {} for name in planner_names}
    for key, value in params.items():
        if "." in key:
            target, field = key.split(".", 1)
            if target not in per_planner:
                raise SystemExit(f"--param {key!r}: planner {target!r} is not being run")
            per_planner[target][field] = value
        elif len(planner_names) == 1:
            per_planner[planner_names[0]][key] = value
        else:
            raise SystemExit(
                f"--param {key!r} is ambiguous with multiple planners; "
                f"prefix it (e.g. cbf.{key}=...)"
            )
    return per_planner


def aggregate(results: list[EpisodeResult]) -> dict:
    survival = np.array([r.survival_time for r in results])
    score = np.array([r.score for r in results])
    causes: dict[str, int] = {}
    for r in results:
        causes[r.cause] = causes.get(r.cause, 0) + 1
    return {
        "episodes": len(results),
        "survival_mean": float(survival.mean()),
        "survival_median": float(np.median(survival)),
        "survival_max": float(survival.max()),
        "timeout_rate": causes.get("timeout", 0) / max(len(results), 1),
        "score_mean": float(score.mean()),
        "score_median": float(np.median(score)),
        "causes": causes,
        "solver_failures": int(sum(r.solver_failures for r in results)),
    }


def format_table(summary: dict[str, dict]) -> str:
    header = (
        f"| {'planner':<8} | {'episodes':>8} | {'survival mean [s]':>17} | "
        f"{'median [s]':>10} | {'timeouts':>8} | {'score mean':>10} | {'deaths (cause)':<28} |"
    )
    sep = "|" + "|".join("-" * (len(col) + 2) for col in [
        "planner ", "episodes", "survival mean [s]", "median [s]", "timeouts", "score mean",
        "deaths (cause)" + " " * 14,
    ]) + "|"
    lines = [header, sep]
    for name, agg in summary.items():
        causes = ", ".join(
            f"{cause}: {n}" for cause, n in sorted(agg["causes"].items()) if cause != "timeout"
        ) or "-"
        lines.append(
            f"| {name:<8} | {agg['episodes']:>8} | {agg['survival_mean']:>17.1f} | "
            f"{agg['survival_median']:>10.1f} | {agg['timeout_rate']:>7.0%} | "
            f"{agg['score_mean']:>10.1f} | {causes:<28} |"
        )
    return "\n".join(lines)


def evaluate(
    planner_names: list[str],
    episodes: int,
    base_seed: int,
    max_time: float,
    backend: Backend,
    out_dir: str | Path,
    render: bool = False,
    params: dict[str, str] | None = None,
    viz=None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_planner_params = split_params(params or {}, planner_names)

    all_results: dict[str, list[EpisodeResult]] = {}
    for name in planner_names:
        results: list[EpisodeResult] = []
        for episode in range(episodes):
            seed = base_seed + episode
            planner = make_planner(name, seed=seed, params=per_planner_params[name])
            result = run_episode(backend, planner, name, seed, max_time, viz=viz)
            results.append(result)
            print(
                f"[{name} {episode + 1}/{episodes} seed={seed}] "
                f"{result.survival_time:.1f}s score={result.score:.0f} cause={result.cause}"
            )
        all_results[name] = results
        if render and results:
            from slither_bot.sim.render import render_h_min, render_trajectory

            representative = sorted(results, key=lambda r: r.survival_time)[len(results) // 2]
            render_trajectory(representative, out_dir / f"traj_{name}.png")
            render_h_min(representative, out_dir / f"hmin_{name}.png")

    summary = {name: aggregate(results) for name, results in all_results.items()}
    table = format_table(summary)
    print()
    print(table)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "config": {
            "planners": planner_names,
            "episodes": episodes,
            "base_seed": base_seed,
            "max_time": max_time,
            "params": params or {},
        },
        "summary": summary,
        "episodes_detail": {
            name: [r.to_json_dict() for r in results] for name, results in all_results.items()
        },
    }
    json_path = out_dir / f"eval_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    (out_dir / "eval_table.md").write_text(table + "\n")
    print(f"\nwrote {json_path} and {out_dir / 'eval_table.md'}")
    return payload


def cmd_eval(args) -> int:
    planner_names = [p.strip() for p in args.planners.split(",") if p.strip()]
    for name in planner_names:
        if name not in PLANNER_NAMES:
            raise SystemExit(f"unknown planner {name!r}; valid: {PLANNER_NAMES}")
    backend = make_backend(args)
    viz = make_viz(args)
    try:
        evaluate(
            planner_names,
            episodes=args.episodes,
            base_seed=args.seed,
            max_time=args.max_time,
            backend=backend,
            out_dir=args.out,
            render=args.render,
            params=parse_params(args.param),
            viz=viz,
        )
    finally:
        if viz is not None:
            viz.close()
        backend.close()
    return 0
