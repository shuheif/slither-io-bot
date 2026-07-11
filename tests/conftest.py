import numpy as np
import pytest

from slither_bot.core import Percept, Snake, empty_body, empty_foods


def make_snake(head, heading=0.0, speed=100.0, radius=10.0, body=None) -> Snake:
    body = empty_body() if body is None else np.asarray(body, dtype=float).reshape(-1, 2)
    return Snake(
        head=np.asarray(head, dtype=float),
        heading=float(heading),
        speed=float(speed),
        radius=float(radius),
        body=body,
    )


def make_percept(
    head=(0.0, 0.0),
    heading=0.0,
    speed=100.0,
    radius=10.0,
    enemies=(),
    foods=(),
    arena_center=None,
    arena_radius=None,
    t=0.0,
    alive=True,
    score=0.0,
) -> Percept:
    foods = np.asarray(foods, dtype=float).reshape(-1, 3) if len(foods) else empty_foods()
    return Percept(
        t=float(t),
        alive=alive,
        self_snake=make_snake(head, heading=heading, speed=speed, radius=radius),
        enemies=list(enemies),
        foods=foods,
        arena_center=None if arena_center is None else np.asarray(arena_center, dtype=float),
        arena_radius=None if arena_radius is None else float(arena_radius),
        score=float(score),
    )


@pytest.fixture
def snake_factory():
    return make_snake


@pytest.fixture
def percept_factory():
    return make_percept


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
