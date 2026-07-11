"""Opt-in integration test: drive LiveBackend against a local fake slither page.

Exercises the real Selenium plumbing (driver startup, PLAY click, READ_STATE /
STEER / BOOST round-trips, pacing) with no network access.  Opt in with::

    SLITHER_BOT_IT=1 python -m pytest tests/test_live_integration.py -v

optionally pointing SLITHER_BOT_CHROME_BINARY / SLITHER_BOT_CHROMEDRIVER at a
local Chrome/chromedriver pair first.
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("SLITHER_BOT_IT"),
    reason="integration test; set SLITHER_BOT_IT=1 (needs Chrome)",
)

FAKE_PAGE = (Path(__file__).parent / "fixtures" / "fake_slither.html").resolve()


def test_live_backend_full_loop_against_fake_page():
    from slither_bot.core import Action
    from slither_bot.live.backend import LiveBackend

    backend = LiveBackend(url=FAKE_PAGE.as_uri(), hz=10.0, headless=True, play_timeout=20.0)
    try:
        percept = backend.reset()
        assert percept.alive
        # The game keeps moving while reset() polls, so allow drift from spawn.
        assert percept.self_snake.head == pytest.approx([21600.0, 21600.0], abs=600.0)
        assert percept.arena_radius == pytest.approx(21600.0)
        assert len(percept.enemies) == 1  # fake page: one enemy besides self
        assert percept.enemies[0].body.shape[0] >= 3
        assert percept.foods.shape == (2, 3)  # null slot skipped

        # Steer straight down (+y in the world frame) and confirm motion.
        start = percept.self_snake.head.copy()
        for _ in range(8):
            percept = backend.step(Action(heading=np.pi / 2))
        moved = percept.self_snake.head - start
        assert moved[1] > 50.0, f"snake should move +y, moved {moved}"
        assert abs(moved[0]) < 0.3 * moved[1]

        # Boost stays off by default (allow_boost=False).
        assert backend.driver.execute_script("return window.boostState") == 0

        # t is relative to episode start and advances with the wall clock.
        assert 0.0 < percept.t < 10.0
    finally:
        backend.close()


def test_probe_against_fake_page(tmp_path, capsys):
    from types import SimpleNamespace

    from slither_bot.live.probe import cmd_probe

    args = SimpleNamespace(
        url=FAKE_PAGE.as_uri(),
        out=str(tmp_path),
        fixture=str(tmp_path / "live_dump.json"),
    )
    rc = cmd_probe(args)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "probe PASSED" in out
    assert (tmp_path / "probe_report.json").exists()
    assert (tmp_path / "live_dump.json").exists()
