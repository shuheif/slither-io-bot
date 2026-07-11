"""Live side-window: what the bot sees and what the planner computes.

Head-centric view in the world frame (y-down, like the game):

* enemy bodies as thick segments colored by **barrier clearance**
  (single-hue sequential: dark red = touching, fading out with distance);
* food as hollow orange dots, the current target as a star;
* the arena wall (dashed) when it is in range;
* the CBF **feasible-heading ring**: 360 sampled headings from the arc
  projection, green (large dot) = executable-safe at full speed, vermilion
  (small dot) = would violate a barrier;
* four direction arrows — executed heading (black), nominal food direction
  (gray), the QP answer ``u`` (purple, length ∝ |u|/v: short = the QP wanted
  to brake), and the commanded heading (blue);
* a monospace readout of t, score, mode, h_min, slack, and constraint counts.

Works over any backend (sim or live) since both emit the same Percept; the
planner internals are read from the planner's ``debug`` dict and every key is
optional, so random/APF episodes simply show fewer layers.  On a headless
machine matplotlib falls back to Agg and updates become invisible no-ops.
"""

from __future__ import annotations

import numpy as np

from slither_bot.core import Action, Percept
from slither_bot.geometry import segment_distances

# Okabe-Ito hues (colorblind-safe); status pair carries a size difference too.
COLORS = {
    "executed": "#000000",
    "nominal": "#8f8f8f",
    "qp": "#CC79A7",
    "command": "#0072B2",
    "food": "#E69F00",
    "feasible": "#009E73",
    "infeasible": "#D55E00",
    "enemy_head": "#67000d",
}


class LiveViz:
    def __init__(
        self,
        update_every: int = 2,
        window_radii: float = 25.0,
        title: str = "slither_bot — planner view",
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Circle

        self._plt = plt
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7.2, 7.2), num=title)
        self.ax.set_aspect("equal")
        self.window_radii = window_radii
        self.update_every = max(1, int(update_every))
        self._tick = 0
        self._dead = False

        self.enemy_lines = LineCollection([], cmap="Reds_r", linewidths=4, zorder=2)
        self.ax.add_collection(self.enemy_lines)
        self.enemy_heads = self.ax.scatter(
            [], [], s=70, marker="X", c=COLORS["enemy_head"], zorder=3
        )
        self.food = self.ax.scatter(
            [], [], s=22, marker="o", facecolors="none", edgecolors=COLORS["food"], zorder=2
        )
        (self.goal_marker,) = self.ax.plot(
            [], [], marker="*", ms=15, color=COLORS["food"], ls="", zorder=4
        )
        self.arcs = self.ax.scatter([], [], s=4, zorder=1)
        (self.self_dot,) = self.ax.plot([0], [0], "o", color="black", ms=9, zorder=5)

        self._arrows = {}
        for key, lw, label in [
            ("executed", 1.5, "executed heading"),
            ("nominal", 1.5, "nominal (to food)"),
            ("qp", 2.2, "QP u (len = |u|/v)"),
            ("command", 2.8, "commanded heading"),
        ]:
            (line,) = self.ax.plot(
                [], [], color=COLORS[key], lw=lw, label=label,
                zorder=6, solid_capstyle="round",
            )
            self._arrows[key] = line

        self.wall = Circle(
            (0.0, 0.0), 1.0, fill=False, ec=COLORS["infeasible"],
            ls="--", lw=1.5, visible=False, zorder=2,
        )
        self.ax.add_patch(self.wall)
        self.text = self.ax.text(
            0.02, 0.02, "", transform=self.ax.transAxes, va="bottom",
            fontsize=9, family="monospace", color="#333333", zorder=7,
        )
        # LineCollection/scatter legend swatches don't reflect the per-datum
        # colors, so the scene entries use explicit proxy handles.
        from matplotlib.lines import Line2D

        proxies = [
            Line2D([], [], color="#a50f15", lw=4, label="enemy body (dark = close)"),
            Line2D([], [], color=COLORS["enemy_head"], marker="X", ls="", ms=8,
                   label="enemy head"),
            Line2D([], [], color=COLORS["food"], marker="o", mfc="none", ls="", ms=6,
                   label="food"),
            Line2D([], [], color=COLORS["food"], marker="*", ls="", ms=10, label="target"),
            Line2D([], [], color=COLORS["feasible"], marker="o", ls="", ms=6,
                   label="feasible heading"),
            Line2D([], [], color=COLORS["infeasible"], marker="o", ls="", ms=3,
                   label="barrier-violating heading"),
        ]
        self.ax.legend(
            handles=proxies + list(self._arrows.values()),
            loc="upper right", fontsize=8, framealpha=0.9,
        )
        self.ax.tick_params(labelsize=8)
        try:
            self.fig.show()
        except Exception:
            pass  # non-GUI backend: updates still work, just invisible

    def update(self, percept: Percept, action: Action | None, debug: dict | None) -> None:
        if self._dead:
            return
        self._tick += 1
        if self._tick % self.update_every:
            return
        if not self._plt.fignum_exists(self.fig.number):
            self._dead = True  # user closed the window: stop updating, keep playing
            return
        debug = debug or {}
        s = percept.self_snake
        p = s.head
        r = max(s.radius, 1e-6)
        v = s.speed if s.speed > 0 else 10.0 * r
        win = self.window_radii * r
        arrow_len = 6.0 * r

        # --- enemy segments colored by clearance (dist - r_self - r_enemy)
        segments, clearances, heads = [], [], []
        for enemy in percept.enemies:
            body = enemy.body - p
            if len(body) == 1:
                body = np.vstack([body, body + 1e-3])
            a, b = body[:-1], body[1:]
            dist, _, _ = segment_distances(np.zeros(2), a, b)
            for i in range(len(a)):
                segments.append([a[i], b[i]])
                clearances.append(dist[i] - r - enemy.radius)
            heads.append(enemy.head - p)
        self.enemy_lines.set_segments(segments)
        if segments:
            self.enemy_lines.set_array(np.asarray(clearances))
            self.enemy_lines.set_clim(0.0, 20.0 * r)
        self.enemy_heads.set_offsets(np.asarray(heads).reshape(-1, 2))

        foods = percept.foods
        self.food.set_offsets(foods[:, :2] - p if len(foods) else np.zeros((0, 2)))

        goal = debug.get("goal")
        if goal is not None:
            g = np.asarray(goal, dtype=float) - p
            self.goal_marker.set_data([g[0]], [g[1]])
        else:
            self.goal_marker.set_data([], [])

        # --- feasible-heading ring from the CBF arc projection
        thetas = debug.get("arc_thetas")
        margins = debug.get("arc_margins")
        if thetas is not None and margins is not None and len(thetas):
            ring_r = 7.5 * r
            offsets = np.stack([ring_r * np.cos(thetas), ring_r * np.sin(thetas)], axis=1)
            feasible = np.asarray(margins) >= 0.0
            self.arcs.set_offsets(offsets)
            self.arcs.set_color(
                np.where(feasible, COLORS["feasible"], COLORS["infeasible"]).tolist()
            )
            self.arcs.set_sizes(np.where(feasible, 9.0, 3.0))
        else:
            self.arcs.set_offsets(np.zeros((0, 2)))

        # --- direction arrows
        def set_arrow(key: str, theta: float | None, frac: float = 1.0) -> None:
            line = self._arrows[key]
            if theta is None:
                line.set_data([], [])
                return
            line.set_data(
                [0.0, frac * arrow_len * np.cos(theta)],
                [0.0, frac * arrow_len * np.sin(theta)],
            )

        set_arrow("executed", s.heading)
        set_arrow("nominal", debug.get("u_nom_heading"))
        u = debug.get("u")
        if u is not None and np.linalg.norm(u) > 1e-9:
            set_arrow(
                "qp", float(np.arctan2(u[1], u[0])), frac=min(float(np.linalg.norm(u)) / v, 1.0)
            )
        else:
            set_arrow("qp", None)
        command = action.heading if action is not None else debug.get("heading")
        set_arrow("command", command)

        # --- arena wall (matplotlib clips the huge circle to the axes)
        if percept.arena_center is not None and percept.arena_radius is not None:
            c = percept.arena_center - p
            self.wall.set_center((float(c[0]), float(c[1])))
            self.wall.set_radius(float(percept.arena_radius))
            self.wall.set_visible(True)
        else:
            self.wall.set_visible(False)

        h_min = debug.get("h_min")
        self.text.set_text(
            f"t={percept.t:8.1f}s  score={percept.score:5.0f}\n"
            f"mode={str(debug.get('mode', '-')):<16} "
            f"h_min={(h_min if h_min is not None else float('nan')):8.1f}\n"
            f"slack={float(debug.get('slack', 0.0)):7.2f}  "
            f"obstacles={str(debug.get('n_obstacles', '-')):>4}  "
            f"rows={str(debug.get('n_constraints', '-')):>4}\n"
            f"solver_failures={debug.get('solver_failures', 0)}"
        )

        self.ax.set_xlim(-win, win)
        self.ax.set_ylim(win, -win)  # y-down, like the game
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            self._dead = True

    def close(self) -> None:
        try:
            self._plt.close(self.fig)
        except Exception:
            pass
