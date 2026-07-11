"""Drive one planner through one episode on any backend, collecting metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from slither_bot.core import Backend, Percept, Planner
from slither_bot.planners import make_planner


@dataclass
class EpisodeResult:
    planner: str
    seed: int
    survival_time: float
    score: float
    cause: str  # "enemy" | "wall" | "timeout" | backend-specific
    ticks: int
    trajectory: np.ndarray  # (T, 2) head positions
    tick_times: list[float] = field(default_factory=list)
    h_min_series: list[float | None] = field(default_factory=list)
    mode_counts: dict[str, int] = field(default_factory=dict)
    solver_failures: int = 0
    arena_center: np.ndarray | None = None
    arena_radius: float | None = None

    def to_json_dict(self) -> dict:
        return {
            "planner": self.planner,
            "seed": self.seed,
            "survival_time": self.survival_time,
            "score": self.score,
            "cause": self.cause,
            "ticks": self.ticks,
            "mode_counts": self.mode_counts,
            "solver_failures": self.solver_failures,
        }


def run_episode(
    backend: Backend,
    planner: Planner,
    planner_name: str,
    seed: int,
    max_time: float = 300.0,
) -> EpisodeResult:
    percept: Percept = backend.reset(seed)
    t0 = percept.t
    trajectory = [percept.self_snake.head.copy()]
    tick_times: list[float] = []
    h_min_series: list[float | None] = []
    mode_counts: dict[str, int] = {}
    arena_center = None if percept.arena_center is None else percept.arena_center.copy()
    arena_radius = percept.arena_radius

    while percept.alive and percept.t - t0 < max_time:
        action = planner.plan(percept)
        percept = backend.step(action)
        trajectory.append(percept.self_snake.head.copy())
        tick_times.append(percept.t - t0)
        debug = getattr(planner, "debug", {}) or {}
        h_min_series.append(debug.get("h_min"))
        mode = debug.get("mode")
        if mode is not None:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

    cause = "timeout" if percept.alive else (getattr(backend, "death_cause", None) or "died")
    debug = getattr(planner, "debug", {}) or {}
    return EpisodeResult(
        planner=planner_name,
        seed=seed,
        survival_time=percept.t - t0,
        score=percept.score,
        cause=cause,
        ticks=len(tick_times),
        trajectory=np.asarray(trajectory),
        tick_times=tick_times,
        h_min_series=h_min_series,
        mode_counts=mode_counts,
        solver_failures=int(debug.get("solver_failures", 0)),
        arena_center=arena_center,
        arena_radius=arena_radius,
    )


def parse_params(pairs: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param expects K=V, got {pair!r}")
        key, value = pair.split("=", 1)
        params[key.strip()] = value.strip()
    return params


def make_backend(args) -> Backend:
    """Backend factory; live/vision are imported lazily so sim runs never
    touch selenium/opencv."""
    if args.backend == "sim":
        from slither_bot.sim.backend import SimBackend
        from slither_bot.sim.world import SimConfig

        return SimBackend(SimConfig(dt=1.0 / args.hz), seed=args.seed)
    if args.backend == "live":
        from slither_bot.live.backend import LiveBackend

        return LiveBackend(hz=args.hz)
    if args.backend == "vision":
        from slither_bot.live.vision_backend import VisionBackend

        return VisionBackend(hz=args.hz)
    raise SystemExit(f"unknown backend {args.backend!r}")


def cmd_run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = parse_params(args.param)

    backend = make_backend(args)
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            planner = make_planner(args.planner, seed=seed, params=params)
            result = run_episode(backend, planner, args.planner, seed, args.max_time)
            print(
                f"[{args.planner} seed={seed}] survived {result.survival_time:.1f}s, "
                f"score={result.score:.0f}, cause={result.cause}, ticks={result.ticks}"
            )
            if result.solver_failures:
                print(f"  solver failures: {result.solver_failures}")
            if args.render:
                from slither_bot.sim.render import render_h_min, render_trajectory

                png = render_trajectory(result, out / f"traj_{args.planner}_{seed}.png")
                print(f"  wrote {png}")
                h_png = render_h_min(result, out / f"hmin_{args.planner}_{seed}.png")
                if h_png is not None:
                    print(f"  wrote {h_png}")
    finally:
        backend.close()
    return 0
