"""Control Barrier Function planner — the centerpiece of the project.

Model: the head as a single integrator ``p_dot = u`` with a box bound
``|ux|, |uy| <= v``.  For every nearby enemy body segment the barrier

    h_i(p) = dist(p, seg_i) - (r_self + r_enemy + margin)      (>= 0 is safe)

must satisfy the CBF condition ``h_dot_i >= -gamma h_i``.  Head-adjacent
enemy segments translate with the enemy head, so for those
``h_dot_i = grad_h_i . (u - v_obs,i)``; deeper body segments are static.
The circular arena wall contributes one more barrier of the same shape.

Safety filtering is a tiny QP over ``z = [ux, uy, delta]``::

    minimize    1/2 |u - u_nom|^2 + 1/2 rho delta^2
    subject to  grad_h_i . u + delta >= gamma h_i_dot_term      (CBF rows)
                |ux|, |uy| <= v                                 (box)
                delta >= 0                                      (slack)

where ``u_nom`` points at the shared food target at speed ``v``.  The single
shared slack ``delta`` keeps the QP feasible even when surrounded; ``rho``
is large so it only activates when the hard constraints genuinely conflict.

Constant-speed execution caveat: the game can only steer, never brake, so a
QP answer that mostly *slows down* (small ``|u*|``, e.g. a symmetric head-on
wall) cannot be executed.  When ``|u*| < exec_min_frac * v`` the planner
switches to the evasive heading along the gradient of the most critical
barrier — "if you cannot brake, turn away as hard as possible".  The same
direction is used if the solver ever fails outright.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slither_bot.core import Action, Percept
from slither_bot.geometry import boundary_barrier
from slither_bot.planners.obstacles import build_obstacles
from slither_bot.planners.qp import solve_qp
from slither_bot.planners.targeting import FoodTargeting, nominal_velocity


@dataclass
class CBFConfig:
    margin_radii: float = 1.0  # inflation absorbing model mismatch, in self-radii
    head_margin_extra_radii: float = 0.5  # extra caution near enemy heads
    d_influence_radii: float = 12.0  # constraint activation clearance
    gamma: float = 1.5  # obstacle CBF rate, 1/s (braking distance ~ v/gamma)
    gamma_boundary: float = 1.0  # softer, earlier wall avoidance
    rho: float = 1e3  # slack penalty
    k_max: int = 40  # constraint cap
    stride: int = 2  # body polyline decimation
    head_pts: int = 6  # body points treated as moving with the enemy head
    exec_min_frac: float = 0.15  # below this |u*|/v, steer evasively instead
    speed_guess_radii: float = 10.0  # v fallback when the backend can't observe speed


class CBFPlanner:
    def __init__(self, config: CBFConfig | None = None) -> None:
        self.cfg = config or CBFConfig()
        self.targeting = FoodTargeting()
        self.debug: dict = {}
        self._solver_failures = 0
        self._last_heading: float | None = None

    def plan(self, percept: Percept) -> Action:
        cfg = self.cfg
        s = percept.self_snake
        p, r = s.head, s.radius
        v = s.speed if s.speed > 0 else cfg.speed_guess_radii * r
        margin = cfg.margin_radii * r

        goal = self.targeting.select(percept)
        u_nom = nominal_velocity(percept, goal, speed=v)

        obstacles = build_obstacles(
            percept,
            d_influence=cfg.d_influence_radii * r,
            k_max=cfg.k_max,
            stride=cfg.stride,
            head_pts=cfg.head_pts,
        )

        # --- stack constraint rows: G z <= h_vec over z = [ux, uy, delta] ---
        rows_G: list[np.ndarray] = []
        rows_h: list[float] = []
        h_values: list[float] = []
        grads: list[np.ndarray] = []

        if len(obstacles):
            extra = np.where(obstacles.is_head, cfg.head_margin_extra_radii * r, 0.0)
            h_obs = obstacles.dist - (r + obstacles.radius + margin + extra)
            for h_i, g_i, v_i in zip(h_obs, obstacles.grad, obstacles.vel):
                # grad . (u - v_obs) >= -gamma h   ->   [-gx, -gy, -1] z <= gamma h - grad . v_obs
                rows_G.append(np.array([-g_i[0], -g_i[1], -1.0]))
                rows_h.append(cfg.gamma * h_i - float(g_i @ v_i))
                h_values.append(float(h_i))
                grads.append(g_i)

        if percept.arena_center is not None and percept.arena_radius is not None:
            h_wall, g_wall = boundary_barrier(p, percept.arena_center, percept.arena_radius)
            h_wall -= r + margin
            if np.linalg.norm(g_wall) > 0:
                rows_G.append(np.array([-g_wall[0], -g_wall[1], -1.0]))
                rows_h.append(cfg.gamma_boundary * h_wall)
                h_values.append(float(h_wall))
                grads.append(g_wall)

        for sign in (1.0, -1.0):  # box: |ux|, |uy| <= v (no slack on these)
            rows_G.append(np.array([sign, 0.0, 0.0]))
            rows_h.append(v)
            rows_G.append(np.array([0.0, sign, 0.0]))
            rows_h.append(v)
        rows_G.append(np.array([0.0, 0.0, -1.0]))  # delta >= 0
        rows_h.append(0.0)

        P = np.diag([1.0, 1.0, cfg.rho])
        q = np.array([-u_nom[0], -u_nom[1], 0.0])
        z = solve_qp(P, q, np.vstack(rows_G), np.array(rows_h))

        mode = "qp"
        slack = 0.0
        if z is None:
            self._solver_failures += 1
            mode = "solver_fallback"
            u = self._evasive_direction(u_nom, h_values, grads) * v
        else:
            u = z[:2]
            slack = float(z[2])
            if np.linalg.norm(u) < cfg.exec_min_frac * v and h_values:
                # The QP wants to brake, but the snake cannot: turn away instead.
                mode = "evasive"
                u = self._evasive_direction(u_nom, h_values, grads) * v

        if np.linalg.norm(u) < 1e-9:
            heading = self._last_heading if self._last_heading is not None else s.heading
        else:
            heading = float(np.arctan2(u[1], u[0]))
        self._last_heading = heading

        self.debug = {
            "mode": mode,
            "u": u.copy(),
            "h_min": min(h_values) if h_values else None,
            "slack": slack,
            "n_constraints": len(rows_h),
            "n_obstacles": len(obstacles),
            "goal": goal,
            "u_nom_heading": float(np.arctan2(u_nom[1], u_nom[0])),
            "heading": heading,
            "solver_failures": self._solver_failures,
        }
        return Action(heading=heading)

    @staticmethod
    def _evasive_direction(
        u_nom: np.ndarray, h_values: list[float], grads: list[np.ndarray]
    ) -> np.ndarray:
        """Unit direction of maximum clearance gain: the gradient of the most
        critical (smallest-h) barrier; the nominal direction if none exist."""
        if not h_values:
            n = np.linalg.norm(u_nom)
            return u_nom / n if n > 1e-9 else np.array([1.0, 0.0])
        g = grads[int(np.argmin(h_values))]
        n = np.linalg.norm(g)
        return g / n if n > 1e-9 else np.array([1.0, 0.0])
