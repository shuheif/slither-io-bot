"""Behavioral tests for the fixed APF planner and the shared targeting."""

import numpy as np
import pytest

from conftest import make_percept, make_snake
from slither_bot.planners.apf import APFPlanner
from slither_bot.planners.targeting import FoodTargeting


def vertical_wall(x, y_min=-80.0, y_max=80.0, step=10.0, radius=5.0):
    ys = np.arange(y_min, y_max + step, step)
    body = np.stack([np.full_like(ys, x), ys], axis=1)
    return make_snake(head=body[0], speed=0.0, radius=radius, body=body)


class TestAPFPlanner:
    def test_points_at_food(self):
        # Regression for the 2022 sign bug, which steered AWAY from food.
        percept = make_percept(head=(0, 0), foods=[(100.0, 0.0, 10.0)])
        action = APFPlanner().plan(percept)
        assert action.heading == pytest.approx(0.0, abs=1e-9)

    def test_points_at_food_any_direction(self):
        percept = make_percept(head=(0, 0), foods=[(-50.0, -50.0, 10.0)])
        action = APFPlanner().plan(percept)
        assert action.heading == pytest.approx(-3 * np.pi / 4, abs=1e-9)

    def test_obstacle_deflects_the_path(self):
        free = make_percept(head=(0, 0), foods=[(200.0, 30.0, 10.0)])
        theta_free = APFPlanner().plan(free).heading
        blocked = make_percept(
            head=(0, 0), foods=[(200.0, 30.0, 10.0)], enemies=[vertical_wall(x=40.0)]
        )
        theta_blocked = APFPlanner().plan(blocked).heading
        assert abs(theta_blocked) > abs(theta_free) + 0.05

    def test_repulsion_is_exactly_zero_beyond_influence(self):
        # Clearance 200 - 10 - 5 = 185 wu > d_influence = 120 wu.
        far = make_percept(
            head=(0, 0), foods=[(100.0, 50.0, 10.0)], enemies=[vertical_wall(x=200.0)]
        )
        free = make_percept(head=(0, 0), foods=[(100.0, 50.0, 10.0)])
        assert APFPlanner().plan(far).heading == APFPlanner().plan(free).heading

    def test_wall_term_pushes_inward(self):
        percept = make_percept(
            head=(480.0, 0.0),
            heading=0.0,
            arena_center=(0.0, 0.0),
            arena_radius=500.0,
        )
        action = APFPlanner().plan(percept)
        assert np.cos(action.heading) < 0.0  # inward


class TestFoodTargeting:
    def test_picks_best_value_per_distance(self):
        targeting = FoodTargeting()
        percept = make_percept(
            head=(0, 0), foods=[(100.0, 0.0, 10.0), (0.0, 110.0, 10.0)]
        )
        goal = targeting.select(percept)
        assert goal == pytest.approx([100.0, 0.0])

    def test_hysteresis_keeps_slightly_worse_current_target(self):
        targeting = FoodTargeting()
        first = make_percept(head=(0, 0), foods=[(100.0, 0.0, 10.0), (0.0, 110.0, 10.0)])
        assert targeting.select(first) == pytest.approx([100.0, 0.0])
        # Challenger now scores ~10% better: within the 20% hysteresis band.
        second = make_percept(head=(0, 0), foods=[(100.0, 0.0, 10.0), (0.0, 91.0, 10.0)])
        assert targeting.select(second) == pytest.approx([100.0, 0.0])

    def test_switches_when_challenger_clearly_better(self):
        targeting = FoodTargeting()
        first = make_percept(head=(0, 0), foods=[(100.0, 0.0, 10.0), (0.0, 110.0, 10.0)])
        assert targeting.select(first) == pytest.approx([100.0, 0.0])
        second = make_percept(head=(0, 0), foods=[(100.0, 0.0, 10.0), (0.0, 60.0, 10.0)])
        assert targeting.select(second) == pytest.approx([0.0, 60.0])

    def test_no_food_returns_none(self):
        assert FoodTargeting().select(make_percept(head=(0, 0))) is None
