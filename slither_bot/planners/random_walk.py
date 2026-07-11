"""The 2022 baseline: pick a uniformly random heading about once a second.

The original bot resampled at its 1 Hz decision rate; at the new 10-15 Hz
loop rate a per-tick resample would just jitter in place, so the heading is
held for ``hold_time`` seconds to stay faithful to the old behavior.
"""

from __future__ import annotations

import numpy as np

from slither_bot.core import Action, Percept


class RandomPlanner:
    def __init__(self, seed: int = 0, hold_time: float = 1.0) -> None:
        self._rng = np.random.default_rng(seed)
        self.hold_time = float(hold_time)
        self._heading = 0.0
        self._next_resample = -np.inf
        self._last_t = np.inf
        self.debug: dict = {}

    def plan(self, percept: Percept) -> Action:
        if percept.t < self._last_t or percept.t >= self._next_resample:
            self._heading = float(self._rng.uniform(-np.pi, np.pi))
            self._next_resample = percept.t + self.hold_time
        self._last_t = percept.t
        self.debug = {"next_resample": self._next_resample}
        return Action(heading=self._heading)
