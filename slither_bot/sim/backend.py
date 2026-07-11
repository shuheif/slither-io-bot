"""Backend wrapper exposing SimWorld through the shared Percept/Action interface."""

from __future__ import annotations

from slither_bot.core import Action, Percept
from slither_bot.sim.world import SimConfig, SimWorld


class SimBackend:
    def __init__(self, config: SimConfig | None = None, seed: int = 0) -> None:
        self.cfg = config or SimConfig()
        self._seed = seed
        self.world: SimWorld | None = None

    @property
    def dt(self) -> float:
        return self.cfg.dt

    @property
    def death_cause(self) -> str | None:
        return self.world.death_cause if self.world is not None else None

    def reset(self, seed: int | None = None) -> Percept:
        if seed is not None:
            self._seed = seed
        self.world = SimWorld(self.cfg, self._seed)
        return self.world.percept()

    def step(self, action: Action) -> Percept:
        assert self.world is not None, "call reset() before step()"
        self.world.step(action)
        return self.world.percept()

    def close(self) -> None:
        pass
