"""Artificial Potential Field planner — the fixed and completed 2022 baseline.

Gradient-descent on U = U_att + U_rep:

* bounded attraction toward the shared food target:
  ``F_att = k_att (g - p)`` within ``d_sat``, saturated at ``k_att d_sat`` beyond;
* the standard repulsive gradient per nearby enemy segment (Khatib-style),
  ``F_rep = eta (1/d - 1/d_inf) (1/d^2) * grad_d``  for clearance ``d < d_inf``,
  where ``grad_d`` points away from the segment;
* the same-form term for the arena wall.

The legacy implementation had the attraction sign inverted (it climbed the
potential, steering *away* from food) and mixed coordinate frames; this is a
reimplementation, not a port.  APF's classic pathologies — local minima and
oscillation between competing gradients — are deliberately left in: they are
the point of comparison against the CBF planner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slither_bot.core import Action, Percept
from slither_bot.geometry import boundary_barrier
from slither_bot.planners.obstacles import build_obstacles
from slither_bot.planners.targeting import FoodTargeting, nominal_velocity


@dataclass
class APFConfig:
    k_att: float = 1.0
    d_att_sat_radii: float = 5.0  # attraction saturates beyond this many self-radii
    d_influence_radii: float = 12.0  # repulsion active inside this clearance
    d_min_radii: float = 0.1  # clearance floor to keep 1/d^2 finite
    k_max: int = 40
    stride: int = 2


class APFPlanner:
    def __init__(self, config: APFConfig | None = None) -> None:
        self.cfg = config or APFConfig()
        self.targeting = FoodTargeting()
        self.debug: dict = {}

    def _eta(self, r: float, d_inf: float) -> float:
        """Size eta so repulsion equals max attraction at clearance 2 r_self."""
        d_ref = 2.0 * r
        rep_shape = (1.0 / d_ref - 1.0 / d_inf) / d_ref**2
        f_att_max = self.cfg.k_att * self.cfg.d_att_sat_radii * r
        return f_att_max / rep_shape

    def plan(self, percept: Percept) -> Action:
        cfg = self.cfg
        s = percept.self_snake
        p, r = s.head, s.radius
        d_inf = cfg.d_influence_radii * r
        d_sat = cfg.d_att_sat_radii * r
        d_min = cfg.d_min_radii * r
        eta = self._eta(r, d_inf)

        goal = self.targeting.select(percept)
        f_att = cfg.k_att * nominal_velocity(percept, goal, speed=d_sat)
        # nominal_velocity returns a vector of magnitude d_sat toward the goal
        # (or along the current heading): exactly the saturated attraction.
        if goal is not None:
            gap = np.linalg.norm(goal - p)
            if gap < d_sat:
                f_att = cfg.k_att * (goal - p)

        obstacles = build_obstacles(
            percept, d_influence=d_inf, k_max=cfg.k_max, stride=cfg.stride
        )
        f_rep = np.zeros(2)
        if len(obstacles):
            d = np.maximum(obstacles.clearance, d_min)
            active = d < d_inf
            if np.any(active):
                mag = eta * (1.0 / d[active] - 1.0 / d_inf) / d[active] ** 2
                f_rep = (mag[:, None] * obstacles.grad[active]).sum(axis=0)

        f_wall = np.zeros(2)
        if percept.arena_center is not None and percept.arena_radius is not None:
            h_wall, grad_wall = boundary_barrier(p, percept.arena_center, percept.arena_radius)
            d_wall = max(h_wall - r, d_min)
            if d_wall < d_inf:
                f_wall = eta * (1.0 / d_wall - 1.0 / d_inf) / d_wall**2 * grad_wall

        f_total = f_att + f_rep + f_wall
        if np.linalg.norm(f_total) < 1e-9:
            heading = s.heading  # local minimum with a perfectly zero gradient
        else:
            heading = float(np.arctan2(f_total[1], f_total[0]))

        self.debug = {
            "goal": goal,
            "f_att": f_att,
            "f_rep_norm": float(np.linalg.norm(f_rep)),
            "n_obstacles": len(obstacles),
            "heading": heading,
        }
        return Action(heading=heading)
