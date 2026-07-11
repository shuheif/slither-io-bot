"""Behavioral tests for the CBF-QP planner on canned percepts."""

import time

import numpy as np
import pytest

from conftest import make_percept, make_snake
from slither_bot.geometry import wrap_angle
from slither_bot.planners.obstacles import build_obstacles
from slither_bot.planners.cbf import CBFConfig, CBFPlanner


def vertical_wall(x, y_min=-80.0, y_max=80.0, step=10.0, radius=5.0, speed=0.0):
    ys = np.arange(y_min, y_max + step, step)
    body = np.stack([np.full_like(ys, x), ys], axis=1)
    return make_snake(head=body[0], heading=0.0, speed=speed, radius=radius, body=body)


def heading_unit(theta):
    return np.array([np.cos(theta), np.sin(theta)])


class TestCBFPlanner:
    def test_no_obstacles_passes_nominal_through(self):
        percept = make_percept(head=(0, 0), foods=[(100.0, 0.0, 10.0)])
        planner = CBFPlanner()
        action = planner.plan(percept)
        assert planner.debug["mode"] == "qp"
        assert action.heading == pytest.approx(0.0, abs=1e-6)

    def test_wall_between_head_and_food_causes_veer(self):
        wall = vertical_wall(x=60.0)
        percept = make_percept(head=(0, 0), foods=[(200.0, 30.0, 10.0)], enemies=[wall])
        planner = CBFPlanner()
        action = planner.plan(percept)
        theta_nom = planner.debug["u_nom_heading"]
        assert abs(action.heading) > abs(theta_nom) + 0.05

        # Verify the CBF condition holds at the chosen u for every constraint
        # the planner saw: grad . (u - v_obs) >= -gamma h - slack.  (The raw
        # undecimated wall would be ~0.5 wu/s stricter; that residual is what
        # the safety margin absorbs, per the decimation design note.)
        cfg = planner.cfg
        u = planner.debug["u"]
        slack = planner.debug["slack"]
        obs = build_obstacles(
            percept,
            d_influence=cfg.d_influence_radii * 10.0,
            k_max=cfg.k_max,
            stride=cfg.stride,
            head_pts=cfg.head_pts,
        )
        extra = np.where(obs.is_head, cfg.head_margin_extra_radii * 10.0, 0.0)
        h = obs.dist - (10.0 + obs.radius + cfg.margin_radii * 10.0 + extra)
        lhs = np.einsum("ij,j->i", obs.grad, u) - np.einsum("ij,ij->i", obs.grad, obs.vel)
        assert len(h) > 0
        assert np.all(lhs >= -cfg.gamma * h - slack - 1e-6)

    def test_approaching_enemy_head_is_treated_more_cautiously_than_static(self):
        theta = {}
        for name, speed in [("static", 0.0), ("moving", 100.0)]:
            body = np.array([[80.0, 0.0], [90.0, 0.0], [100.0, 0.0], [110.0, 0.0]])
            enemy = make_snake(
                head=body[0], heading=np.pi, speed=speed, radius=5.0, body=body
            )
            percept = make_percept(
                head=(0, 0), foods=[(200.0, 30.0, 10.0)], enemies=[enemy]
            )
            planner = CBFPlanner()
            action = planner.plan(percept)
            theta[name] = abs(wrap_angle(action.heading - planner.debug["u_nom_heading"]))
        assert theta["moving"] > theta["static"] + 0.1

    def test_surrounded_activates_slack_without_crashing(self):
        ring = [
            make_snake(head=20.0 * heading_unit(phi), speed=0.0, radius=5.0)
            for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ]
        percept = make_percept(head=(0, 0), foods=[(100.0, 0.0, 10.0)], enemies=ring)
        planner = CBFPlanner()
        action = planner.plan(percept)
        assert np.isfinite(action.heading)
        assert planner.debug["slack"] > 0.0
        assert planner.debug["h_min"] < 0.0

    def test_boundary_turns_an_outward_heading_inward(self):
        percept = make_percept(
            head=(480.0, 0.0), heading=0.0, arena_center=(0.0, 0.0), arena_radius=500.0
        )
        planner = CBFPlanner()
        action = planner.plan(percept)
        outward = np.array([1.0, 0.0])
        assert heading_unit(action.heading) @ outward < 0.0  # commanded inward

    def test_constraint_cap_and_runtime_with_many_segments(self):
        rng = np.random.default_rng(7)
        enemies = []
        for k in range(20):
            phi = 2 * np.pi * k / 20
            start = 40.0 * heading_unit(phi)
            direction = heading_unit(phi + rng.uniform(-0.5, 0.5))
            body = start[None, :] + np.arange(51)[:, None] * 4.0 * direction[None, :]
            enemies.append(make_snake(head=body[0], speed=50.0, radius=5.0, body=body))
        percept = make_percept(head=(0, 0), foods=[(100.0, 0.0, 10.0)], enemies=enemies)
        planner = CBFPlanner()

        durations = []
        for _ in range(10):
            t0 = time.perf_counter()
            action = planner.plan(percept)
            durations.append(time.perf_counter() - t0)
        cfg = planner.cfg
        assert planner.debug["n_constraints"] <= cfg.k_max + 5
        assert np.isfinite(action.heading)
        assert sorted(durations)[len(durations) // 2] < 0.05  # median < 50 ms

    def test_solver_failure_falls_back_to_max_clearance(self, monkeypatch):
        monkeypatch.setattr("slither_bot.planners.cbf.solve_qp", lambda *a, **k: None)
        wall = vertical_wall(x=60.0)
        percept = make_percept(head=(0, 0), foods=[(200.0, 0.0, 10.0)], enemies=[wall])
        planner = CBFPlanner()
        action = planner.plan(percept)
        assert planner.debug["mode"] == "solver_fallback"
        assert planner.debug["solver_failures"] == 1
        assert np.isfinite(action.heading)
        # Max-clearance direction for a wall ahead (+x) is away from it (-x).
        assert heading_unit(action.heading) @ np.array([1.0, 0.0]) < 0.0

    def test_config_units_scale_with_radius(self):
        cfg = CBFConfig(margin_radii=2.0)
        planner = CBFPlanner(cfg)
        percept = make_percept(head=(0, 0), radius=20.0, foods=[(300.0, 0.0, 10.0)])
        planner.plan(percept)  # smoke: parameters derived from radius, no crash
        assert planner.debug["mode"] == "qp"
