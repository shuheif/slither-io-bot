"""Planners: random walk, artificial potential fields, control barrier functions."""

from __future__ import annotations

from dataclasses import fields

from slither_bot.planners.apf import APFConfig, APFPlanner
from slither_bot.planners.cbf import CBFConfig, CBFPlanner
from slither_bot.planners.random_walk import RandomPlanner

PLANNER_NAMES = ("random", "apf", "cbf")


def _apply_params(config, params: dict[str, str]):
    valid = [f.name for f in fields(config)]
    for key, value in params.items():
        if key not in valid:
            raise ValueError(f"unknown parameter {key!r}; valid: {sorted(valid)}")
        current = getattr(config, key)
        setattr(config, key, type(current)(value))
    return config


def make_planner(name: str, seed: int = 0, params: dict[str, str] | None = None):
    """Build a fresh planner (planners hold per-episode state; make one per episode)."""
    params = dict(params or {})
    if name == "random":
        hold_time = float(params.pop("hold_time", 1.0))
        if params:
            raise ValueError(f"unknown parameter(s) for random: {sorted(params)}")
        return RandomPlanner(seed=seed, hold_time=hold_time)
    if name == "apf":
        return APFPlanner(_apply_params(APFConfig(), params))
    if name == "cbf":
        return CBFPlanner(_apply_params(CBFConfig(), params))
    raise ValueError(f"unknown planner {name!r}; valid: {PLANNER_NAMES}")
