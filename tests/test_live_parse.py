"""Offline tests of the live-state parser on a synthetic READ_STATE dump.

When the probe has been run on a real machine, its dumped
``tests/fixtures/live_dump.json`` is picked up here too, so the parser is
additionally exercised against the real client's schema.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from slither_bot.live import js_bridge
from slither_bot.live.js_bridge import RADIUS_PER_SC, SPEED_SCALE, parse_state

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def raw():
    return json.loads((FIXTURES / "synthetic_dump.json").read_text())


class TestParseState:
    def test_self_snake_fields(self, raw):
        percept = parse_state(raw)
        assert percept.alive
        assert percept.self_snake.head == pytest.approx([21000.0, 20500.0])
        assert percept.self_snake.heading == pytest.approx(1.2)
        assert percept.self_snake.speed == pytest.approx(5.78 * SPEED_SCALE)
        assert percept.self_snake.radius == pytest.approx(1.0 * RADIUS_PER_SC)
        assert percept.score == 12

    def test_arena_from_grd(self, raw):
        percept = parse_state(raw)
        assert percept.arena_center == pytest.approx([21600.0, 21600.0])
        assert percept.arena_radius == pytest.approx(21600.0)

    def test_null_gap_splits_enemy_into_runs(self, raw):
        percept = parse_state(raw)
        # enemy 1 -> two runs (head run + detached static run), enemy 2 -> one
        assert len(percept.enemies) == 3
        head_run, detached, head_only = percept.enemies

        assert head_run.head == pytest.approx([21200.0, 20600.0])
        assert head_run.heading == pytest.approx(-2.0)
        assert head_run.speed == pytest.approx(5.9 * SPEED_SCALE)
        assert head_run.body.shape == (5, 2)

        assert detached.speed == 0.0  # deep body: static obstacle
        assert detached.body.shape == (3, 2)
        assert detached.head == pytest.approx([21500.0, 20900.0])

    def test_enemy_without_pts_becomes_point_obstacle(self, raw):
        percept = parse_state(raw)
        head_only = percept.enemies[2]
        assert head_only.body.shape == (1, 2)
        assert head_only.body[0] == pytest.approx([20800.0, 20450.0])

    def test_foods_shape_and_values(self, raw):
        percept = parse_state(raw)
        assert percept.foods.shape == (4, 3)
        assert percept.foods[2] == pytest.approx([21100.0, 20550.0, 8.0])

    def test_time_override_and_scales(self, raw):
        percept = parse_state(raw, speed_scale=1.0, radius_per_sc=2.0, t=3.5)
        assert percept.t == pytest.approx(3.5)
        assert percept.self_snake.speed == pytest.approx(5.78)
        assert percept.self_snake.radius == pytest.approx(2.0)

    def test_not_playing_is_dead(self):
        percept = parse_state({"playing": False})
        assert not percept.alive
        assert percept.enemies == []
        assert len(percept.foods) == 0
        assert parse_state({}).alive is False
        assert parse_state(None).alive is False

    def test_snake_missing_is_dead(self):
        # In-game but the state globals are renamed: unusable percept.
        assert parse_state({"playing": True, "snake_missing": True}).alive is False

    def test_planner_accepts_parsed_percept(self, raw):
        from slither_bot.planners.cbf import CBFPlanner

        planner = CBFPlanner()
        action = planner.plan(parse_state(raw))
        assert np.isfinite(action.heading)

    def test_real_dump_if_present(self):
        real = FIXTURES / "live_dump.json"
        if not real.exists():
            pytest.skip("no live_dump.json yet — run `python -m slither_bot probe` on a Mac")
        percept = parse_state(json.loads(real.read_text()))
        assert percept.alive
        assert np.isfinite(percept.self_snake.head).all()


class TestGlobalNameTemplating:
    def test_snippets_are_fully_rendered(self):
        for js in (js_bridge.READ_STATE, js_bridge.PROBE, js_bridge.DIAGNOSE_MENU):
            assert "__SNAKE__" not in js
            assert "__SNAKES__" not in js
            assert "__FOODS__" not in js
        assert 'window["snake"]' in js_bridge.READ_STATE
        assert 'window["snakes"]' in js_bridge.READ_STATE
        assert 'window["foods"]' in js_bridge.READ_STATE

    def test_set_global_names_round_trip(self):
        try:
            js_bridge.set_global_names(snakes="slithers", foods="fud")
            assert js_bridge.GLOBALS["snakes"] == "slithers"
            assert 'window["slithers"]' in js_bridge.READ_STATE
            assert 'window["fud"]' in js_bridge.READ_STATE
            assert 'window["slithers"]' in js_bridge.PROBE
        finally:
            js_bridge.set_global_names(snake="snake", snakes="snakes", foods="foods")
        assert 'window["snakes"]' in js_bridge.READ_STATE

    def test_none_values_keep_current_names(self):
        before = dict(js_bridge.GLOBALS)
        js_bridge.set_global_names(snake=None, snakes=None)
        assert js_bridge.GLOBALS == before

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            js_bridge.set_global_names(bogus="x")


class TestLiveBackendOffline:
    def test_constructs_without_browser(self):
        from slither_bot.live.backend import LiveBackend

        backend = LiveBackend(hz=10.0)
        assert backend.dt == pytest.approx(0.1)
        assert backend.driver is None
        backend.close()  # no-op when never opened

    def test_cli_parses_live_and_probe_args(self):
        from slither_bot.cli import build_parser

        run_args = build_parser().parse_args(
            ["run", "--backend", "live", "--planner", "cbf", "--boost"]
        )
        assert run_args.backend == "live" and run_args.boost

        probe_args = build_parser().parse_args(["probe", "--url", "http://slither.io"])
        assert probe_args.url == "http://slither.io"
