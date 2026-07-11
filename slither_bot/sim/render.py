"""Headless-safe matplotlib rendering of episode artifacts (PNG files only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def render_trajectory(result, path: str | Path) -> Path:
    """Trajectory plot for one episode: arena, path colored by time, death marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    if result.arena_radius is not None:
        wall = plt.Circle(result.arena_center, result.arena_radius, fill=False, color="0.4")
        ax.add_patch(wall)
        lim = 1.05 * result.arena_radius
        ax.set_xlim(result.arena_center[0] - lim, result.arena_center[0] + lim)
        ax.set_ylim(result.arena_center[1] - lim, result.arena_center[1] + lim)

    traj = np.asarray(result.trajectory)
    if len(traj) > 1:
        points = ax.scatter(
            traj[:, 0], traj[:, 1], c=np.arange(len(traj)), cmap="viridis", s=4
        )
        fig.colorbar(points, ax=ax, label="tick", shrink=0.8)
    ax.plot(*traj[0], marker="o", color="tab:green", markersize=9, label="start")
    if result.cause != "timeout":
        ax.plot(*traj[-1], marker="x", color="tab:red", markersize=12, mew=3, label=result.cause)

    ax.invert_yaxis()  # world frame is y-down; render it the way the game looks
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title(
        f"{result.planner} seed={result.seed}: {result.survival_time:.1f}s, "
        f"score={result.score:.0f}, cause={result.cause}"
    )
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def render_h_min(result, path: str | Path) -> Path | None:
    """Minimum barrier value over time for a CBF episode; skipped if never logged."""
    series = [(t, h) for t, h in zip(result.tick_times, result.h_min_series) if h is not None]
    if not series:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts, hs = zip(*series)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(ts, hs, lw=1.0)
    ax.axhline(0.0, color="tab:red", lw=1, ls="--", label="h = 0 (safety boundary)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("min barrier h [wu]")
    ax.set_title(f"{result.planner} seed={result.seed}: closest approach over time")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
