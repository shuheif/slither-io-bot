"""Core contract shared by every backend and planner.

Frame and unit conventions (normative for the whole project)
------------------------------------------------------------
* Positions are in the backend's **absolute world frame**, x to the right and
  **y down** — the same orientation slither.io's client uses for its world
  coordinates (``.xx``/``.yy``), so live values pass through untransformed.
* Angles are radians in ``(-pi, pi]`` measured with ``atan2(dy, dx)``.
  Because y points down, positive angles rotate clockwise on screen; a
  heading of 0 points along +x.
* Distances are in the backend's world units ("wu"), time in seconds,
  speeds in wu/s.  Planners never assume a particular scale: their
  parameters are expressed in multiples of the bot's own body radius (and
  seconds) and converted via ``Percept.self_snake.radius`` at plan time, so
  the same planner runs unchanged in the simulator and the live game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Snake:
    """One snake, ours or an enemy's."""

    head: np.ndarray  # (2,) world position, wu
    heading: float  # rad
    speed: float  # wu/s; 0.0 when the backend cannot observe it
    radius: float  # collision radius, wu
    body: np.ndarray  # (M, 2) body points, index 0 nearest the head; (0, 2) allowed


@dataclass
class Percept:
    """Everything a planner is allowed to know at one control tick."""

    t: float  # seconds since episode start
    alive: bool
    self_snake: Snake
    enemies: list[Snake]
    foods: np.ndarray  # (K, 3): x, y, size; (0, 3) allowed
    arena_center: np.ndarray | None  # None => boundary unknown, barrier skipped
    arena_radius: float | None
    score: float  # length proxy; sim: exact, live: DOM/body-count proxy


@dataclass
class Action:
    """What a planner is allowed to command."""

    heading: float  # commanded world heading, rad
    boost: bool = False


class Planner(Protocol):
    """plan() maps a Percept to an Action; ``debug`` carries per-tick internals
    (e.g. h_min, slack, active target) for logging and plots."""

    debug: dict

    def plan(self, percept: Percept) -> Action: ...


class Backend(Protocol):
    """A game the planner can play: the simulator or the live browser."""

    dt: float  # nominal seconds between step() returns

    def reset(self) -> Percept: ...

    def step(self, action: Action) -> Percept: ...

    def close(self) -> None: ...


def empty_foods() -> np.ndarray:
    return np.zeros((0, 3))


def empty_body() -> np.ndarray:
    return np.zeros((0, 2))
