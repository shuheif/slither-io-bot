"""A small slither-like world for offline testing and tuning.

Deliberately mirrors the parts of slither.io that matter to the planners:

* the bot moves at constant speed and can only steer, with a bounded turn
  rate — the *real* dynamics, so the CBF's single-integrator model mismatch
  shows up here during tuning rather than only in the live game;
* enemies are moving snakes (random-waypoint walkers) whose trailing bodies
  are the obstacles; touching any enemy body point or the arena wall kills;
* food pellets respawn so density stays constant.

Everything is pure numpy and seeded: the same seed reproduces an episode
bit-for-bit (given the same action sequence).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slither_bot.core import Action, Percept, Snake, empty_body
from slither_bot.geometry import segment_distances, wrap_angle


@dataclass
class SimConfig:
    arena_radius: float = 1000.0  # wu
    dt: float = 1.0 / 15.0  # s per tick
    self_speed: float = 100.0  # wu/s
    self_radius: float = 10.0  # wu
    turn_rate_max: float = 3.0  # rad/s (the model mismatch the CBF must survive)
    boost_factor: float = 2.0  # speed multiplier while Action.boost
    n_enemies: int = 6
    enemy_speed: float = 90.0
    enemy_radius: float = 8.0
    enemy_body_points: int = 30
    enemy_point_spacing: float = 12.0  # wu between recorded body points
    enemy_turn_rate_max: float = 2.0
    enemy_waypoint_reached: float = 60.0  # wu
    enemy_retarget_mean_s: float = 8.0  # mean seconds between random retargets
    enemy_wall_retreat_frac: float = 0.85  # retarget toward center beyond this radius
    n_foods: int = 120
    food_size_min: float = 1.0
    food_size_max: float = 10.0
    eat_radius_factor: float = 1.5  # eat when food center within factor * r_self
    view_radius: float = 600.0  # percept culling range


class _SimEnemy:
    def __init__(self, head: np.ndarray, heading: float, cfg: SimConfig, rng) -> None:
        self.head = head.astype(float)
        self.heading = float(heading)
        self.waypoint = head.copy()
        self._dist_since_point = 0.0
        back = -np.array([np.cos(heading), np.sin(heading)])
        self.trail = [
            head + (i + 1) * cfg.enemy_point_spacing * back
            for i in range(cfg.enemy_body_points - 1)
        ]

    def body_array(self) -> np.ndarray:
        return np.vstack([self.head[None, :], np.asarray(self.trail)])

    def step(self, cfg: SimConfig, rng) -> None:
        retarget = (
            np.linalg.norm(self.waypoint - self.head) < cfg.enemy_waypoint_reached
            or rng.uniform() < cfg.dt / cfg.enemy_retarget_mean_s
        )
        if np.linalg.norm(self.head) > cfg.enemy_wall_retreat_frac * cfg.arena_radius:
            self.waypoint = _sample_in_disk(rng, 0.3 * cfg.arena_radius)
        elif retarget:
            self.waypoint = _sample_in_disk(rng, 0.8 * cfg.arena_radius)

        to_wp = self.waypoint - self.head
        desired = np.arctan2(to_wp[1], to_wp[0])
        max_turn = cfg.enemy_turn_rate_max * cfg.dt
        self.heading = wrap_angle(
            self.heading + np.clip(wrap_angle(desired - self.heading), -max_turn, max_turn)
        )
        move = cfg.enemy_speed * cfg.dt
        self.head = self.head + move * np.array([np.cos(self.heading), np.sin(self.heading)])
        self._dist_since_point += move
        if self._dist_since_point >= cfg.enemy_point_spacing:
            self.trail.insert(0, self.head.copy())
            del self.trail[-1]
            self._dist_since_point = 0.0


def _sample_in_disk(rng, radius: float, avoid: np.ndarray | None = None, min_dist: float = 0.0):
    while True:
        p = rng.uniform(-radius, radius, size=2)
        if np.linalg.norm(p) > radius:
            continue
        if avoid is not None and np.linalg.norm(p - avoid) < min_dist:
            continue
        return p


class SimWorld:
    def __init__(self, cfg: SimConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.alive = True
        self.death_cause: str | None = None
        self.score = 0.0

        self.p = _sample_in_disk(self.rng, 0.3 * cfg.arena_radius)
        self.heading = float(self.rng.uniform(-np.pi, np.pi))

        self.enemies = [
            _SimEnemy(
                _sample_in_disk(self.rng, 0.8 * cfg.arena_radius, avoid=self.p, min_dist=200.0),
                self.rng.uniform(-np.pi, np.pi),
                cfg,
                self.rng,
            )
            for _ in range(cfg.n_enemies)
        ]

        positions = np.array(
            [_sample_in_disk(self.rng, 0.95 * cfg.arena_radius) for _ in range(cfg.n_foods)]
        )
        sizes = self.rng.uniform(cfg.food_size_min, cfg.food_size_max, size=cfg.n_foods)
        self.foods = np.column_stack([positions, sizes])

    def step(self, action: Action) -> None:
        cfg = self.cfg
        if self.alive:
            max_turn = cfg.turn_rate_max * cfg.dt
            self.heading = wrap_angle(
                self.heading
                + np.clip(wrap_angle(action.heading - self.heading), -max_turn, max_turn)
            )
            speed = cfg.self_speed * (cfg.boost_factor if action.boost else 1.0)
            self.p = self.p + speed * cfg.dt * np.array(
                [np.cos(self.heading), np.sin(self.heading)]
            )

            for enemy in self.enemies:
                enemy.step(cfg, self.rng)

            d_food = np.linalg.norm(self.foods[:, :2] - self.p, axis=1)
            for idx in np.flatnonzero(d_food < cfg.eat_radius_factor * cfg.self_radius):
                self.score += 1.0
                self.foods[idx, :2] = _sample_in_disk(self.rng, 0.95 * cfg.arena_radius)
                self.foods[idx, 2] = self.rng.uniform(cfg.food_size_min, cfg.food_size_max)

            self._check_death()
        self.t += cfg.dt

    def _check_death(self) -> None:
        cfg = self.cfg
        if np.linalg.norm(self.p) + cfg.self_radius >= cfg.arena_radius:
            self.alive = False
            self.death_cause = "wall"
            return
        kill_dist = cfg.self_radius + cfg.enemy_radius
        for enemy in self.enemies:
            body = enemy.body_array()
            if np.linalg.norm(enemy.head - self.p) > kill_dist + len(body) * cfg.enemy_point_spacing:
                continue  # cheap bound: whole body is out of reach
            dist, _, _ = segment_distances(self.p, body[:-1], body[1:])
            if dist.min() < kill_dist:
                self.alive = False
                self.death_cause = "enemy"
                return

    def percept(self) -> Percept:
        cfg = self.cfg
        enemies = []
        for enemy in self.enemies:
            body = enemy.body_array()
            if np.linalg.norm(body - self.p, axis=1).min() < cfg.view_radius:
                enemies.append(
                    Snake(
                        head=enemy.head.copy(),
                        heading=enemy.heading,
                        speed=cfg.enemy_speed,
                        radius=cfg.enemy_radius,
                        body=body,
                    )
                )
        in_view = np.linalg.norm(self.foods[:, :2] - self.p, axis=1) < cfg.view_radius
        return Percept(
            t=self.t,
            alive=self.alive,
            self_snake=Snake(
                head=self.p.copy(),
                heading=self.heading,
                speed=cfg.self_speed,
                radius=cfg.self_radius,
                body=empty_body(),
            ),
            enemies=enemies,
            foods=self.foods[in_view].copy(),
            arena_center=np.zeros(2),
            arena_radius=cfg.arena_radius,
            score=self.score,
        )
