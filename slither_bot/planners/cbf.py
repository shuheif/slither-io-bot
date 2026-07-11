"""Control Barrier Function planner — the centerpiece of the project.

Stage 1, CBF-QP safety filter.  Model the head as a single integrator
``p_dot = u`` with box bound ``|ux|, |uy| <= v``.  For every nearby enemy
body segment the barrier

    h_i(p) = dist(p, seg_i) - (r_self + r_enemy + margin)      (>= 0 is safe)

must satisfy ``h_dot_i >= -gamma h_i``.  Head-adjacent enemy segments
translate with the enemy head, so there ``h_dot_i = grad_h_i . (u - v_obs,i)``;
deeper body segments are static (they trace the head's old path).  The
circular arena wall adds one more row of the same shape.  Filtering is a
tiny QP over ``z = [ux, uy, delta]``::

    minimize    1/2 |u - u_nom|^2 + 1/2 rho delta^2
    subject to  grad_h_i . u + delta >= -gamma h_i + grad_h_i . v_obs,i
                |ux|, |uy| <= v
                delta >= 0

with ``u_nom`` pointing at the shared food target at speed ``v`` and a
single shared slack ``delta`` that keeps the QP feasible when surrounded.

Stage 2, constant-speed feasibility projection.  The game can only steer:
whatever we command, the executed control is ``u_exec = v e(theta)`` at full
speed.  A QP answer that *brakes* (small ``|u*|``) is therefore not
executable — followed naively it drills into obstacles at full speed.  But
at fixed speed each CBF row becomes an *allowed arc* of headings:

    grad_h_i . (v e(theta) - v_obs,i) >= -gamma h_i
        <=>  cos(theta - phi_i) >= b_i / v,   phi_i = angle(grad_h_i)

so the executable-safe heading set is an intersection of arcs.  We command
the heading inside that set closest to the QP's direction (the QP acts as
the preference/tie-break); if the set is empty, the heading maximizing the
worst constraint margin — turn as favorably as possible when trapped.  The
same projection with the nominal direction serves as the fallback if the
solver ever fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slither_bot.core import Action, Percept
from slither_bot.geometry import boundary_barrier, wrap_angle, wrap_angles as _wrap_all
from slither_bot.planners.obstacles import build_obstacles
from slither_bot.planners.qp import solve_qp
from slither_bot.planners.targeting import FoodTargeting, nominal_velocity


@dataclass
class CBFConfig:
    margin_radii: float = 1.5  # inflation absorbing model mismatch, in self-radii
    head_margin_extra_radii: float = 1.0  # extra caution near enemy heads
    d_influence_radii: float = 20.0  # activation clearance (sized for mutual approach)
    gamma: float = 1.0  # obstacle CBF rate, 1/s
    gamma_boundary: float = 1.0  # wall avoidance rate
    rho: float = 1e3  # slack penalty
    k_max: int = 40  # constraint cap
    stride: int = 2  # body polyline decimation
    head_pts: int = 6  # body points treated as moving with the enemy head
    pref_min_frac: float = 0.3  # |u*|/v below which the nominal direction is preferred
    arc_samples: int = 360  # heading discretization for the feasibility projection
    beta_last_cmd: float = 0.5  # hysteresis: penalty weight on leaving the last command
    h_urgent_radii: float = 3.0  # below this h_min, escape from the CURRENT heading
    speed_guess_radii: float = 10.0  # v fallback when the backend can't observe speed


class CBFPlanner:
    def __init__(self, config: CBFConfig | None = None) -> None:
        self.cfg = config or CBFConfig()
        self.targeting = FoodTargeting()
        self.debug: dict = {}
        self._solver_failures = 0
        self._last_cmd: float | None = None

    def plan(self, percept: Percept) -> Action:
        cfg = self.cfg
        s = percept.self_snake
        p, r = s.head, s.radius
        v = s.speed if s.speed > 0 else cfg.speed_guess_radii * r
        margin = cfg.margin_radii * r

        goal = self.targeting.select(percept)
        u_nom = nominal_velocity(percept, goal, speed=v)
        theta_nom = float(np.arctan2(u_nom[1], u_nom[0]))

        obstacles = build_obstacles(
            percept,
            d_influence=cfg.d_influence_radii * r,
            k_max=cfg.k_max,
            stride=cfg.stride,
            head_pts=cfg.head_pts,
        )

        # --- CBF rows shared by both stages: normals, barrier values, rhs ---
        # Row i demands  grad_i . u >= rhs_i  with  rhs_i = -gamma_i h_i + grad_i . v_obs,i
        normals: list[np.ndarray] = []
        rhs: list[float] = []
        h_values: list[float] = []

        if len(obstacles):
            extra = np.where(obstacles.is_head, cfg.head_margin_extra_radii * r, 0.0)
            h_obs = obstacles.dist - (r + obstacles.radius + margin + extra)
            for h_i, g_i, v_i in zip(h_obs, obstacles.grad, obstacles.vel):
                normals.append(g_i)
                rhs.append(-cfg.gamma * h_i + float(g_i @ v_i))
                h_values.append(float(h_i))

        if percept.arena_center is not None and percept.arena_radius is not None:
            h_wall, g_wall = boundary_barrier(p, percept.arena_center, percept.arena_radius)
            h_wall -= r + margin
            if np.linalg.norm(g_wall) > 0:
                normals.append(g_wall)
                rhs.append(-cfg.gamma_boundary * h_wall)
                h_values.append(float(h_wall))

        # --- stage 1: the QP over z = [ux, uy, delta] ---
        rows_G = [np.array([-g[0], -g[1], -1.0]) for g in normals]
        rows_h = [-b for b in rhs]
        for sign in (1.0, -1.0):  # box |ux|, |uy| <= v (no slack)
            rows_G.append(np.array([sign, 0.0, 0.0]))
            rows_h.append(v)
            rows_G.append(np.array([0.0, sign, 0.0]))
            rows_h.append(v)
        rows_G.append(np.array([0.0, 0.0, -1.0]))  # delta >= 0
        rows_h.append(0.0)

        P = np.diag([1.0, 1.0, cfg.rho])
        q = np.array([-u_nom[0], -u_nom[1], 0.0])
        z = solve_qp(P, q, np.vstack(rows_G), np.array(rows_h))

        if z is None:
            self._solver_failures += 1
            mode = "solver_fallback"
            u = u_nom.copy()
            slack = 0.0
            theta_pref = theta_nom
        else:
            u = z[:2]
            slack = float(z[2])
            mode = "qp"
            if np.linalg.norm(u) >= cfg.pref_min_frac * v:
                theta_pref = float(np.arctan2(u[1], u[0]))
            else:
                theta_pref = theta_nom  # a braking answer carries little direction

        # --- stage 2: project onto the executable-safe heading arcs ---
        h_min = min(h_values) if h_values else np.inf
        urgent = h_min < cfg.h_urgent_radii * r
        heading, pref_was_feasible, arc_thetas, arc_margins = self._project_heading(
            theta_pref, s.heading, v, normals, rhs, urgent
        )
        self._last_cmd = heading
        if mode == "qp" and not pref_was_feasible:
            mode = "projected"

        self.debug = {
            "mode": mode,
            "u": u.copy(),
            "h_min": min(h_values) if h_values else None,
            "slack": slack,
            "n_constraints": len(rows_h),
            "n_obstacles": len(obstacles),
            "goal": goal,
            "u_nom_heading": theta_nom,
            "heading": heading,
            "solver_failures": self._solver_failures,
            "arc_thetas": arc_thetas,  # sampled headings (abs), None when no rows
            "arc_margins": arc_margins,  # worst constraint margin per heading
        }
        return Action(heading=heading)

    def _project_heading(
        self,
        theta_pref: float,
        theta_current: float,
        v: float,
        normals: list[np.ndarray],
        rhs: list[float],
        urgent: bool,
    ) -> tuple[float, bool, np.ndarray | None, np.ndarray | None]:
        """Pick a commanded heading satisfying every arc constraint
        ``v * (g_i . e(theta)) >= rhs_i``; the max-margin heading if none does.

        Among feasible headings the choice is normally the one nearest the
        preference, with a hysteresis penalty for jumping away from the last
        command (a memoryless nearest-arc rule flip-flops between the two
        sides of an obstacle, and each flip costs a slow turn-rate-limited
        swing through the danger zone).  In an *urgent* situation the goal no
        longer matters: pick the feasible heading nearest the currently
        executed one — the escape the snake can actually reach fastest.

        Returns ``(theta, pref_was_feasible, sampled_thetas, worst_margins)``
        — the last two feed the live visualization (None when no rows).
        """
        if not normals:
            return theta_pref, True, None, None

        offsets = np.linspace(-np.pi, np.pi, self.cfg.arc_samples, endpoint=False)
        thetas = theta_pref + offsets
        e = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)  # (S, 2)
        margins = v * (e @ np.asarray(normals).T) - np.asarray(rhs)[None, :]  # (S, N)
        worst = margins.min(axis=1)

        feasible = worst >= 0.0
        zero_idx = self.cfg.arc_samples // 2  # offsets[zero_idx] == 0.0
        pref_ok = bool(feasible[zero_idx])

        if not np.any(feasible):
            return wrap_angle(float(thetas[int(np.argmax(worst))])), False, thetas, worst

        idx = np.flatnonzero(feasible)
        if urgent:
            cost = np.abs(_wrap_all(thetas[idx] - theta_current))
        else:
            cost = np.abs(offsets[idx])
            if self._last_cmd is not None:
                cost = cost + self.cfg.beta_last_cmd * np.abs(
                    _wrap_all(thetas[idx] - self._last_cmd)
                )
        best = idx[int(np.argmin(cost))]
        if pref_ok and not urgent and best == zero_idx:
            return theta_pref, True, thetas, worst
        return wrap_angle(float(thetas[best])), pref_ok, thetas, worst
