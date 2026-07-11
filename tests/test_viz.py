"""Offline (Agg) tests of the live planner-view window."""

import matplotlib
import numpy as np
import pytest

from conftest import make_percept, make_snake
from slither_bot.core import Action


@pytest.fixture
def viz():
    matplotlib.use("Agg", force=True)  # headless-safe regardless of environment
    from slither_bot.viz import LiveViz

    v = LiveViz(update_every=1)
    yield v
    v.close()


def cbf_like_debug():
    thetas = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    return {
        "mode": "qp",
        "u": np.array([50.0, 10.0]),
        "h_min": 42.0,
        "slack": 0.0,
        "n_obstacles": 3,
        "n_constraints": 8,
        "goal": np.array([120.0, 30.0]),
        "u_nom_heading": 0.2,
        "heading": 0.4,
        "solver_failures": 0,
        "arc_thetas": thetas,
        "arc_margins": np.cos(thetas),  # half the ring feasible
    }


def scene_percept():
    wall = make_snake(
        head=(60.0, -40.0), body=[[60.0, -40.0], [60.0, 0.0], [60.0, 40.0]], radius=5.0
    )
    return make_percept(
        head=(0.0, 0.0),
        enemies=[wall],
        foods=[(30.0, 20.0, 5.0)],
        arena_center=(0.0, 0.0),
        arena_radius=500.0,
        score=7.0,
    )


def test_updates_all_layers_and_saves(viz, tmp_path):
    viz.update(scene_percept(), Action(heading=0.4), cbf_like_debug())
    assert len(viz.enemy_lines.get_segments()) == 2
    assert viz.arcs.get_offsets().shape[0] == 360
    assert viz.enemy_heads.get_offsets().shape[0] == 1
    assert viz.wall.get_visible()
    assert "h_min" in viz.text.get_text()
    out = tmp_path / "viz.png"
    viz.fig.savefig(out)
    assert out.stat().st_size > 0


def test_tolerates_minimal_debug_and_none(viz):
    percept = make_percept(head=(0.0, 0.0))
    viz.update(percept, Action(heading=0.0), {})
    viz.update(percept, Action(heading=0.0), None)
    assert len(viz.enemy_lines.get_segments()) == 0
    assert viz.arcs.get_offsets().shape[0] == 0
    assert not viz.wall.get_visible()


def test_update_every_throttles():
    matplotlib.use("Agg", force=True)
    from slither_bot.viz import LiveViz

    v = LiveViz(update_every=3)
    try:
        percept = scene_percept()
        v.update(percept, Action(heading=0.0), cbf_like_debug())
        v.update(percept, Action(heading=0.0), cbf_like_debug())
        assert len(v.enemy_lines.get_segments()) == 0  # ticks 1-2 skipped
        v.update(percept, Action(heading=0.0), cbf_like_debug())
        assert len(v.enemy_lines.get_segments()) == 2  # tick 3 rendered
    finally:
        v.close()


def test_survives_user_closing_the_window(viz):
    viz._plt.close(viz.fig)
    viz.update(scene_percept(), Action(heading=0.0), cbf_like_debug())  # no crash
    assert viz._dead


def test_runner_integration_with_sim(tmp_path):
    matplotlib.use("Agg", force=True)
    from slither_bot.eval.runner import run_episode
    from slither_bot.planners import make_planner
    from slither_bot.sim.backend import SimBackend
    from slither_bot.sim.world import SimConfig
    from slither_bot.viz import LiveViz

    viz = LiveViz(update_every=2)
    try:
        backend = SimBackend(SimConfig(), seed=3)
        planner = make_planner("cbf", seed=3)
        result = run_episode(backend, planner, "cbf", 3, max_time=2.0, viz=viz)
        assert result.ticks > 0
        assert viz._tick == result.ticks  # update() called every control tick
    finally:
        viz.close()
