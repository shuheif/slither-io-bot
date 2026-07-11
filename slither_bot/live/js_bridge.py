"""Every piece of JavaScript the bot injects into slither.io, in one place.

The game keeps its state in globals on ``window`` (client of 2016-2022 era):

===============  ==============================================================
``snake``        our snake: ``.xx/.yy`` world pos, ``.ang`` heading (rad),
                 ``.sp`` speed unit, ``.sc`` scale, ``.sct`` segment count,
                 ``.pts`` body parts (objects with ``.xx/.yy``, ``.dying``)
``snakes``       all snakes in view, same shape (match ``.id`` to skip self)
``foods``        pellets: ``.xx/.yy/.sz`` (array reuses slots -> null guards)
``grd``          map half-size; center ~ (grd, grd), border radius ~ grd
``playing``      false on the menu / after death
``xm``, ``ym``   mouse offset from the window center; the game steers toward
                 ``atan2(ym, xm)`` every frame -> writing them steers the snake
``setAcceleration(n)``  boost on/off
===============  ==============================================================

If the client ever renames these, fix the names HERE (nowhere else touches
them) — ``python -m slither_bot probe`` verifies each one and, on failure,
scans ``window`` for plausible replacements.

Unit conversions (probe-verified, see ``live/probe.py``):
* collision radius  ~= ``RADIUS_PER_SC * sc`` world units;
* speed in wu/s     ~= ``SPEED_SCALE * sp``.

The world frame matches ``core.py``'s convention exactly (x right, y down,
angles from atan2), so positions and angles pass through untransformed.
"""

from __future__ import annotations

import numpy as np

from slither_bot.core import Percept, Snake, empty_body, empty_foods

RADIUS_PER_SC = 14.5  # wu of collision radius per unit of .sc (client draws ~29*sc px wide)
SPEED_SCALE = 32.0  # wu/s per unit of .sp (normal sp ~ 5.8 -> ~185 wu/s)

# --- read the world (one round-trip). arguments[0] = view radius in wu ---
READ_STATE = """
const view = arguments[0];
if (!window.playing || !window.snake) { return {playing: false}; }
const s = window.snake;
const view2 = view * view;
const enemies = [];
for (const sn of (window.snakes || [])) {
  if (!sn || sn.id === s.id) continue;
  const pts = [];
  let lastKept = false;
  for (const p of (sn.pts || [])) {
    if (!p || p.dying) { continue; }
    const dx = p.xx - s.xx, dy = p.yy - s.yy;
    if (dx * dx + dy * dy < view2) {
      pts.push([p.xx, p.yy]);
      lastKept = true;
    } else if (lastKept) {
      pts.push(null);           // gap marker: the polyline left the view
      lastKept = false;
    }
  }
  const hx = sn.xx - s.xx, hy = sn.yy - s.yy;
  if (pts.length || hx * hx + hy * hy < view2) {
    enemies.push({x: sn.xx, y: sn.yy, ang: sn.ang, sp: sn.sp, sc: sn.sc, pts: pts});
  }
}
const foods = [];
for (const f of (window.foods || [])) {
  if (!f) continue;
  const dx = f.xx - s.xx, dy = f.yy - s.yy;
  if (dx * dx + dy * dy < view2) { foods.push([f.xx, f.yy, f.sz || 1]); }
}
return {
  playing: true,
  t: Date.now() / 1000.0,
  self: {x: s.xx, y: s.yy, ang: s.ang, sp: s.sp, sc: s.sc, sct: s.sct || 0},
  enemies: enemies,
  foods: foods,
  grd: window.grd || null,
};
"""

# --- steer: arguments[0] = world heading in radians ---
STEER = """
const theta = arguments[0];
window.xm = Math.round(100 * Math.cos(theta));
window.ym = Math.round(100 * Math.sin(theta));
"""

# --- boost: arguments[0] = 1 or 0 ---
BOOST = """
if (typeof window.setAcceleration === 'function') { window.setAcceleration(arguments[0]); }
"""

# --- start a game from the menu; returns what it managed to do ---
PLAY = """
const nick = document.getElementById('nick');
if (nick && arguments[0]) { nick.value = arguments[0]; }
const btn = document.querySelector('#playh .btnt')
         || document.querySelector('#playh div')
         || document.querySelector('.btnt');
if (btn) { btn.click(); return 'clicked'; }
if (typeof window.connect === 'function' && !window.playing) { window.connect(); return 'connected'; }
return 'no-button';
"""

# --- probe: which globals exist and look right ---
PROBE = """
function kind(v) {
  if (v === undefined || v === null) return null;
  if (Array.isArray(v)) return 'array[' + v.length + ']';
  return typeof v;
}
return {
  snake: kind(window.snake),
  snakes: kind(window.snakes),
  foods: kind(window.foods),
  grd: (typeof window.grd === 'number') ? window.grd : null,
  playing: kind(window.playing),
  xm: kind(window.xm),
  ym: kind(window.ym),
  setAcceleration: kind(window.setAcceleration),
  connect: kind(window.connect),
};
"""

# --- fallback scanner: find window arrays that look like snakes/foods ---
SCAN = """
const found = [];
for (const name of Object.getOwnPropertyNames(window)) {
  try {
    const v = window[name];
    if (!Array.isArray(v) || v.length < 3) continue;
    const item = v.find(x => x && typeof x === 'object');
    if (!item) continue;
    if (typeof item.xx === 'number' && typeof item.yy === 'number') {
      found.push({name: name, length: v.length, keys: Object.keys(item).slice(0, 12)});
    }
  } catch (e) { /* cross-origin or getter trap */ }
}
return found;
"""


def parse_state(
    raw: dict,
    speed_scale: float = SPEED_SCALE,
    radius_per_sc: float = RADIUS_PER_SC,
    t: float | None = None,
) -> Percept:
    """READ_STATE result -> Percept (see core.py for the frame convention).

    Enemy body polylines arrive with ``null`` gap markers where they left the
    view radius; each contiguous run becomes its own obstacle snake.  The run
    containing the true head keeps the enemy's heading and speed; detached
    deeper-body runs are static obstacles (speed 0), which matches how the
    planners treat non-head segments anyway.
    """
    if not raw or not raw.get("playing", False):
        dummy = Snake(head=np.zeros(2), heading=0.0, speed=0.0, radius=1.0, body=empty_body())
        return Percept(
            t=t if t is not None else float(raw.get("t", 0.0) if raw else 0.0),
            alive=False,
            self_snake=dummy,
            enemies=[],
            foods=empty_foods(),
            arena_center=None,
            arena_radius=None,
            score=0.0,
        )

    me = raw["self"]
    self_snake = Snake(
        head=np.array([me["x"], me["y"]], dtype=float),
        heading=float(me["ang"]),
        speed=float(me["sp"]) * speed_scale,
        radius=float(me["sc"]) * radius_per_sc,
        body=empty_body(),
    )

    enemies: list[Snake] = []
    for entry in raw.get("enemies", []):
        radius = float(entry["sc"]) * radius_per_sc
        head = np.array([entry["x"], entry["y"]], dtype=float)
        runs: list[list[list[float]]] = [[]]
        for p in entry.get("pts", []):
            if p is None:
                if runs[-1]:
                    runs.append([])
            else:
                runs[-1].append(p)
        runs = [run for run in runs if run]
        if not runs:
            runs = [[list(head)]]
        for i, run in enumerate(runs):
            body = np.asarray(run, dtype=float)
            if i == 0:
                enemies.append(
                    Snake(
                        head=head,
                        heading=float(entry["ang"]),
                        speed=float(entry["sp"]) * speed_scale,
                        radius=radius,
                        body=body,
                    )
                )
            else:  # detached deeper-body run: a static obstacle
                enemies.append(
                    Snake(head=body[0].copy(), heading=0.0, speed=0.0, radius=radius, body=body)
                )

    foods = raw.get("foods") or []
    foods = np.asarray(foods, dtype=float).reshape(-1, 3) if foods else empty_foods()

    grd = raw.get("grd")
    arena_center = np.array([grd, grd], dtype=float) if grd else None
    arena_radius = float(grd) if grd else None

    return Percept(
        t=t if t is not None else float(raw["t"]),
        alive=True,
        self_snake=self_snake,
        enemies=enemies,
        foods=foods,
        arena_center=arena_center,
        arena_radius=arena_radius,
        score=float(me.get("sct", 0)),
    )
