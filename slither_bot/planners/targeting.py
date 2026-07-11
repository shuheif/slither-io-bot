"""Shared food-target selection.

APF and CBF chase the same nominal goal chosen here, so the comparison
between them isolates the safety mechanism (potential repulsion vs QP
safety filter) rather than differences in greed.

Foods carry no IDs across ticks, so the current target is re-identified by
position; hysteresis keeps the goal from flapping between similar pellets
(which would make both planners oscillate for reasons unrelated to safety).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slither_bot.core import Percept


@dataclass
class TargetingConfig:
    hysteresis: float = 1.2  # challenger must score >=20% better to steal the target
    max_range_radii: float = 60.0  # ignore food farther than this many self-radii


class FoodTargeting:
    def __init__(self, config: TargetingConfig | None = None) -> None:
        self.config = config or TargetingConfig()
        self._target: np.ndarray | None = None

    def select(self, percept: Percept) -> np.ndarray | None:
        """Goal point (2,) in world coordinates, or None when no food is in range."""
        cfg = self.config
        p = percept.self_snake.head
        r = percept.self_snake.radius
        if len(percept.foods) == 0:
            self._target = None
            return None

        pos = percept.foods[:, :2]
        size = percept.foods[:, 2]
        dist = np.linalg.norm(pos - p, axis=1)
        in_range = dist < cfg.max_range_radii * r
        if not np.any(in_range):
            self._target = None
            return None

        score = size / np.maximum(dist, r)  # value per distance
        score[~in_range] = -np.inf
        challenger = int(np.argmax(score))

        if self._target is not None:
            # Re-identify the current target by position (foods have no IDs).
            d_to_old = np.linalg.norm(pos - self._target, axis=1)
            current = int(np.argmin(d_to_old))
            still_exists = d_to_old[current] < 2.0 * r and np.isfinite(score[current])
            if still_exists and score[challenger] < cfg.hysteresis * score[current]:
                self._target = pos[current].copy()
                return self._target

        self._target = pos[challenger].copy()
        return self._target


def nominal_velocity(percept: Percept, goal: np.ndarray | None, speed: float) -> np.ndarray:
    """Desired velocity: toward the goal at ``speed``; hold heading when goalless."""
    s = percept.self_snake
    ahead = np.array([np.cos(s.heading), np.sin(s.heading)])
    if goal is None:
        return speed * ahead
    d = goal - s.head
    n = np.linalg.norm(d)
    if n < 1e-9:
        return speed * ahead
    return speed * d / n
