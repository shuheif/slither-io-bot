"""Simulator determinism, dynamics, and death-cause tests."""

import numpy as np
import pytest

from slither_bot.core import Action
from slither_bot.geometry import wrap_angle
from slither_bot.sim.backend import SimBackend
from slither_bot.sim.world import SimConfig, SimWorld


def scripted_trajectory(seed, ticks=200):
    world = SimWorld(SimConfig(), seed)
    traj = []
    for k in range(ticks):
        world.step(Action(heading=0.3 * np.sin(0.05 * k)))
        traj.append(world.p.copy())
        if not world.alive:
            break
    return np.asarray(traj)


class TestDeterminism:
    def test_same_seed_is_bit_identical(self):
        a = scripted_trajectory(seed=42)
        b = scripted_trajectory(seed=42)
        assert a.shape == b.shape
        assert np.array_equal(a, b)

    def test_different_seed_differs(self):
        a = scripted_trajectory(seed=42)
        b = scripted_trajectory(seed=43)
        assert a.shape != b.shape or not np.array_equal(a, b)


class TestDynamics:
    def test_turn_rate_is_clamped(self):
        cfg = SimConfig(n_enemies=0, arena_radius=1e6)
        world = SimWorld(cfg, seed=1)
        start_heading = world.heading
        target = wrap_angle(start_heading + np.pi)  # demand a full reversal
        ticks = 0
        while abs(wrap_angle(world.heading - target)) > 0.05:
            world.step(Action(heading=target))
            ticks += 1
            assert ticks < 200, "never reached the reversed heading"
        assert ticks * cfg.dt >= np.pi / cfg.turn_rate_max - cfg.dt - 1e-9

    def test_boost_doubles_displacement(self):
        cfg = SimConfig(n_enemies=0, arena_radius=1e6)
        normal = SimWorld(cfg, seed=5)
        boosted = SimWorld(cfg, seed=5)
        p0 = normal.p.copy()
        heading = normal.heading
        normal.step(Action(heading=heading))
        boosted.step(Action(heading=heading, boost=True))
        d_normal = np.linalg.norm(normal.p - p0)
        d_boosted = np.linalg.norm(boosted.p - p0)
        assert d_boosted == pytest.approx(cfg.boost_factor * d_normal)


class TestDeathAndFood:
    def test_wall_death(self):
        cfg = SimConfig(n_enemies=0, arena_radius=200.0)
        backend = SimBackend(cfg, seed=3)
        percept = backend.reset()
        outward = float(np.arctan2(percept.self_snake.head[1], percept.self_snake.head[0]))
        for _ in range(1000):
            percept = backend.step(Action(heading=outward))
            if not percept.alive:
                break
        assert not percept.alive
        assert backend.death_cause == "wall"

    def test_enemy_death(self):
        cfg = SimConfig(n_enemies=1, arena_radius=2000.0, enemy_speed=0.0)
        world = SimWorld(cfg, seed=7)
        target = world.enemies[0].body_array().mean(axis=0)
        for _ in range(5000):
            to_target = target - world.p
            world.step(Action(heading=float(np.arctan2(to_target[1], to_target[0]))))
            if not world.alive:
                break
        assert not world.alive
        assert world.death_cause == "enemy"

    def test_food_respawns_and_scores(self):
        cfg = SimConfig(n_enemies=0, arena_radius=1e5, n_foods=60)
        world = SimWorld(cfg, seed=11)
        for _ in range(5000):
            in_view = world.foods
            d = np.linalg.norm(in_view[:, :2] - world.p, axis=1)
            nearest = in_view[int(np.argmin(d)), :2]
            to_food = nearest - world.p
            world.step(Action(heading=float(np.arctan2(to_food[1], to_food[0]))))
            if world.score > 0:
                break
        assert world.score >= 1.0
        assert len(world.foods) == cfg.n_foods  # eaten pellets respawned


class TestPercept:
    def test_culls_far_enemies_and_food(self):
        cfg = SimConfig(view_radius=150.0, arena_radius=4000.0, n_enemies=8)
        backend = SimBackend(cfg, seed=13)
        percept = backend.reset()
        p = percept.self_snake.head
        for enemy in percept.enemies:
            assert np.linalg.norm(enemy.body - p, axis=1).min() < cfg.view_radius
        if len(percept.foods):
            assert np.linalg.norm(percept.foods[:, :2] - p, axis=1).max() < cfg.view_radius

    def test_reset_same_seed_gives_same_start(self):
        backend = SimBackend(SimConfig(), seed=17)
        p1 = backend.reset().self_snake.head
        p2 = backend.reset().self_snake.head
        assert np.array_equal(p1, p2)
        p3 = backend.reset(seed=18).self_snake.head
        assert not np.array_equal(p1, p3)
