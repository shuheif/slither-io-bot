"""Every piece of JavaScript the bot injects into slither.io, in one place.

The game keeps its state in globals on ``window``:

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

Client builds RENAME the state globals from time to time (a 2026 build was
observed with ``playing``/``xm``/``ym``/``foods``/``grd`` intact but ``snake``
and ``snakes`` gone).  The names live in the ``GLOBALS`` dict below — edit the
values there and every snippet picks them up.  ``python -m slither_bot probe``
detects renames at runtime (a joined game whose snake is invisible), scans
``window`` for the new names, live-verifies them via ``set_global_names()``,
and prints the exact values to put in ``GLOBALS``.

Unit conversions (probe-verified, see ``live/probe.py``):
* collision radius  ~= ``RADIUS_PER_SC * sc`` world units;
* speed in wu/s     ~= ``SPEED_SCALE * sp``.

The world frame matches ``core.py``'s convention exactly (x right, y down,
angles from atan2), so positions and angles pass through untransformed.
"""

from __future__ import annotations

import json

import numpy as np

from slither_bot.core import Percept, Snake, empty_body, empty_foods

RADIUS_PER_SC = 14.5  # wu of collision radius per unit of .sc (client draws ~29*sc px wide)
SPEED_SCALE = 32.0  # wu/s per unit of .sp (normal sp ~ 5.8 -> ~185 wu/s)

# The game-state global names for the current client build.  If the probe
# reports verified renames, update the VALUES here (keys stay fixed).
GLOBALS = {
    "snake": "snake",
    "snakes": "snakes",
    "foods": "foods",
}


def _render(template: str) -> str:
    out = template
    for key, name in GLOBALS.items():
        out = out.replace(f"__{key.upper()}__", json.dumps(name))
    return out


def set_global_names(**names: str | None) -> None:
    """Override state-global names at runtime (e.g. from the probe's scanner)
    and re-render every snippet that references them."""
    for key, value in names.items():
        if key not in GLOBALS:
            raise KeyError(f"unknown global {key!r}; known: {sorted(GLOBALS)}")
        if value:
            GLOBALS[key] = value
    _rebuild()


# --- read the world (one round-trip). arguments[0] = view radius in wu ------
# Distinguishes "on the menu" (playing false) from "in-game but the snake
# global is absent" (snake_missing: the client renamed its state globals).
_READ_STATE_TPL = """
const view = arguments[0];
if (!window.playing) { return {playing: false}; }
const s = window[__SNAKE__];
if (!s) { return {playing: true, snake_missing: true}; }
const view2 = view * view;
const enemies = [];
for (const sn of (window[__SNAKES__] || [])) {
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
for (const f of (window[__FOODS__] || [])) {
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

# --- start a game from the menu ----------------------------------------------
# The client's own Enter-key handler runs `play_btn.elem.onclick()`, whose
# handler latches `want_play`; the main loop then calls `connect()` once the
# server list (`sos`, fetched from /i33628.txt) is ready.  One click is enough:
# while `connecting`/`want_play`/`waiting_for_sos` the client is already busy
# self-retrying (it rotates servers every ~3.3 s), so we report state instead
# of clicking again.  A disabled play_btn only skips the button strategy —
# some builds keep it disabled while connect() works fine (observed 2026).
# Working bots' direct-join fallback is `dead_mtm = 0; login_fr = 0; connect()`
# — it bypasses the button and ad gates entirely and reads the nickname from
# #nick at ws.onopen.
# arguments[0] = nickname, arguments[1] = force (skip buttons, connect() now).
# Returns {strategy, state}.
PLAY = """
function st() {
  return {
    playing: !!window.playing,
    connecting: !!window.connecting,
    connected: !!window.connected,
    want_play: !!window.want_play,
    waiting_for_sos: !!window.waiting_for_sos,
    sos_len: (window.sos && window.sos.length) || 0,
    btn_disabled: !!(window.play_btn && window.play_btn.disabled),
    dead_mtm: (typeof window.dead_mtm === 'number') ? window.dead_mtm : null,
    protocol: location.protocol
  };
}
const nick = document.getElementById('nick');
if (nick && arguments[0]) { nick.value = arguments[0]; }
const force = !!arguments[1];
const s0 = st();
if (s0.playing) { return {strategy: 'already-playing', state: s0}; }
if (!force && (s0.connecting || s0.want_play || s0.waiting_for_sos)) {
  return {strategy: 'join-in-progress', state: s0};
}
const pb = window.play_btn;
if (!force && pb && pb.elem && typeof pb.elem.onclick === 'function' && !pb.disabled) {
  pb.elem.onclick();
  return {strategy: 'play_btn.elem.onclick', state: st()};
}
if (!force && !s0.btn_disabled) {
  const btn = document.querySelector('#playh .btnt') || document.querySelector('#playh div');
  if (btn) {
    for (const type of ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click']) {
      btn.dispatchEvent(new MouseEvent(type, {bubbles: true, cancelable: true, view: window}));
    }
    return {strategy: 'dom-events', state: st()};
  }
}
if (typeof window.connect === 'function') {
  window.dead_mtm = 0;
  window.login_fr = 0;
  window.connect();
  return {strategy: 'connect()', state: st()};
}
return {strategy: 'nothing-applicable', state: s0};
"""

# --- best-effort consent/CMP dismissal (top document only) -------------------
# The mirrored real client ships no CMP, and synthetic JS clicks bypass any
# overlay anyway — this exists for the human manual-join path and for future
# client changes.  Consent-looking iframes are detected and REPORTED only.
DISMISS_CONSENT = """
const out = {clicked: [], present: [], iframes: []};
const SELECTORS = [
  '.fc-consent-root .fc-cta-consent',
  '.fc-consent-root button.fc-primary-button',
  '#qc-cmp2-ui button[mode="primary"]',
  '#onetrust-accept-btn-handler',
  '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
  '.cmpboxbtnyes'
];
for (const sel of SELECTORS) {
  const el = document.querySelector(sel);
  if (!el) continue;
  out.present.push(sel);
  const r = el.getBoundingClientRect();
  if (r.width > 0 && r.height > 0) { el.click(); out.clicked.push(sel); }
}
for (const f of document.querySelectorAll('iframe')) {
  const sig = (f.src || '') + ' ' + (f.id || '') + ' ' + (f.name || '');
  if (/consent|cmp|fundingchoices|sp_msg|privacy/i.test(sig)) {
    out.iframes.push({src: (f.src || '').slice(0, 120), id: f.id || null});
  }
}
return out;
"""

# --- menu diagnosis for the join-failure path --------------------------------
_DIAGNOSE_MENU_TPL = """
function kind(v) {
  if (v === undefined) return 'undefined';
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array[' + v.length + ']';
  return typeof v;
}
const pb = window.play_btn;
const diag = {
  href: location.href,
  protocol: location.protocol,
  readyState: document.readyState,
  globals: {
    snake: kind(window[__SNAKE__]),
    snakes: kind(window[__SNAKES__]),
    foods: kind(window[__FOODS__]),
    play_btn: kind(pb),
    play_btn_elem: pb ? kind(pb.elem) : null,
    play_btn_disabled: pb ? !!pb.disabled : null,
    connect: kind(window.connect),
    forceServer: kind(window.forceServer),
    playing: !!window.playing,
    connecting: !!window.connecting,
    connected: !!window.connected,
    want_play: !!window.want_play,
    waiting_for_sos: !!window.waiting_for_sos,
    sos_len: (window.sos && window.sos.length) || 0,
    bso: (window.bso && window.bso.ip) ? String(window.bso.ip) + ':' + String(window.bso.po) : null,
    dead_mtm: (typeof window.dead_mtm === 'number') ? window.dead_mtm : kind(window.dead_mtm),
    login_fr: (typeof window.login_fr === 'number') ? window.login_fr : kind(window.login_fr),
    shoa: !!window.shoa,
    grd: (typeof window.grd === 'number') ? window.grd : null,
    ws: window.ws ? {readyState: window.ws.readyState, url: String(window.ws.url || '')} : null
  },
  buttons: [],
  center_stack: [],
  consent_iframes: []
};
const seen = new Set();
const candidates = [];
for (const rootSel of ['#login', '#playh']) {
  const root = document.querySelector(rootSel);
  if (root) { candidates.push(...root.querySelectorAll('div,button,a')); }
}
candidates.push(...document.querySelectorAll('.btnt'));
for (const el of candidates) {
  if (diag.buttons.length >= 12 || seen.has(el)) continue;
  seen.add(el);
  const r = el.getBoundingClientRect();
  const text = (el.textContent || '').trim().slice(0, 30);
  if (!text && r.width === 0) continue;
  diag.buttons.push({
    tag: el.tagName.toLowerCase(),
    id: el.id || null,
    cls: (el.className && String(el.className).slice(0, 60)) || null,
    text: text,
    visible: r.width > 0 && r.height > 0,
    rect: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]
  });
}
let at = document.elementFromPoint(Math.round(innerWidth / 2), Math.round(innerHeight / 2));
while (at && diag.center_stack.length < 6) {
  diag.center_stack.push(at.tagName.toLowerCase() + (at.id ? '#' + at.id : ''));
  at = at.parentElement;
}
for (const f of document.querySelectorAll('iframe')) {
  const sig = (f.src || '') + ' ' + (f.id || '') + ' ' + (f.name || '');
  if (/consent|cmp|fundingchoices|sp_msg|privacy/i.test(sig)) {
    diag.consent_iframes.push({src: (f.src || '').slice(0, 120), id: f.id || null});
  }
}
return diag;
"""

# --- probe: which globals exist and look right ---
# kind() distinguishes 'null' (declared, not yet populated — e.g. `snake`
# on the menu) from undefined/missing (returned as null -> MISSING).
_PROBE_TPL = """
function kind(v) {
  if (v === undefined) return null;
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array[' + v.length + ']';
  return typeof v;
}
return {
  snake: kind(window[__SNAKE__]),
  snakes: kind(window[__SNAKES__]),
  foods: kind(window[__FOODS__]),
  grd: (typeof window.grd === 'number') ? window.grd : null,
  playing: kind(window.playing),
  xm: kind(window.xm),
  ym: kind(window.ym),
  setAcceleration: kind(window.setAcceleration),
  connect: kind(window.connect),
};
"""

# --- scanner: find renamed state globals while IN-GAME ----------------------
# Arrays of objects with .xx/.yy are snakes/foods candidates; a plain object
# with .xx/.yy/.ang is a self-snake candidate — strongest when it is an
# identity member of the snakes-candidate array.  Returns a ready suggestion.
SCAN = """
const arrays = [];
const objects = [];
for (const name of Object.getOwnPropertyNames(window)) {
  let v;
  try { v = window[name]; } catch (e) { continue; }
  if (Array.isArray(v)) {
    if (v.length < 1) continue;
    const item = v.find(x => x && typeof x === 'object');
    if (!item || typeof item.xx !== 'number' || typeof item.yy !== 'number') continue;
    arrays.push({
      name: name, length: v.length, keys: Object.keys(item).slice(0, 15),
      has_pts: Array.isArray(item.pts),
      has_ang: typeof item.ang === 'number',
      has_sz: typeof item.sz === 'number' || typeof item.rad === 'number'
    });
  } else if (v && typeof v === 'object') {
    let xx, yy, ang, pts;
    try { xx = v.xx; yy = v.yy; ang = v.ang; pts = v.pts; } catch (e) { continue; }
    if (typeof xx === 'number' && typeof yy === 'number') {
      objects.push({
        name: name, keys: Object.keys(v).slice(0, 15),
        has_ang: typeof ang === 'number', has_pts: Array.isArray(pts)
      });
    }
  }
}
const suggestion = {snake: null, snakes: null, foods: null};
for (const a of arrays) {
  if ((a.has_pts || a.has_ang) && !suggestion.snakes) { suggestion.snakes = a.name; }
  if (a.has_sz && !a.has_ang && !suggestion.foods) { suggestion.foods = a.name; }
}
if (suggestion.snakes) {
  const arr = window[suggestion.snakes];
  for (const o of objects) {
    if (o.has_ang && arr.includes(window[o.name])) { suggestion.snake = o.name; break; }
  }
}
if (!suggestion.snake) {
  const o = objects.find(o => o.has_ang && o.has_pts);
  if (o) { suggestion.snake = o.name; }
}
return {arrays: arrays.slice(0, 20), objects: objects.slice(0, 20), suggestion: suggestion};
"""

READ_STATE = ""
DIAGNOSE_MENU = ""
PROBE = ""


def _rebuild() -> None:
    global READ_STATE, DIAGNOSE_MENU, PROBE
    READ_STATE = _render(_READ_STATE_TPL)
    DIAGNOSE_MENU = _render(_DIAGNOSE_MENU_TPL)
    PROBE = _render(_PROBE_TPL)


_rebuild()


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
    if not raw or not raw.get("playing", False) or raw.get("snake_missing"):
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
