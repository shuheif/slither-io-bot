"""Opt-in integration tests: drive LiveBackend against a local fake slither page.

Exercises the real Selenium plumbing (driver startup, every join strategy,
READ_STATE / STEER / BOOST round-trips, pacing, failure diagnostics) with no
network access.  Opt in with::

    SLITHER_BOT_IT=1 python -m pytest tests/test_live_integration.py -v

optionally pointing SLITHER_BOT_CHROME_BINARY / SLITHER_BOT_CHROMEDRIVER at a
local Chrome/chromedriver pair first.  The fake page implements the real
client's join state machine in several variants (see fake_slither.html).
"""

import json
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


@pytest.fixture(autouse=True)
def _restore_global_names():
    # The probe's rescue path rebinds js_bridge.GLOBALS for the session;
    # keep tests independent of each other.
    yield
    from slither_bot.live import js_bridge

    js_bridge.set_global_names(snake="snake", snakes="snakes", foods="foods")


def page_url(variant: str | None = None) -> str:
    return FAKE_PAGE.as_uri() + (f"?variant={variant}" if variant else "")


def make_backend(variant: str | None = None, **kwargs):
    from slither_bot.live.backend import LiveBackend

    kwargs.setdefault("hz", 10.0)
    kwargs.setdefault("headless", True)
    kwargs.setdefault("play_timeout", 20.0)
    return LiveBackend(url=page_url(variant), **kwargs)


def join_strategies(backend) -> list[str]:
    return [entry.get("strategy") for entry in backend.last_join_log]


def test_live_backend_full_loop_against_fake_page():
    from slither_bot.core import Action

    backend = make_backend()
    try:
        percept = backend.reset()
        assert percept.alive
        assert "play_btn.elem.onclick" in join_strategies(backend)
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


def test_join_via_dom_events_when_no_play_btn_object():
    backend = make_backend("domclick")
    try:
        assert backend.reset().alive
        assert "dom-events" in join_strategies(backend)
    finally:
        backend.close()


def test_join_via_direct_connect_when_no_button():
    backend = make_backend("nobutton")
    try:
        assert backend.reset().alive
        assert "connect()" in join_strategies(backend)
    finally:
        backend.close()


def test_consent_overlay_is_dismissed_and_join_succeeds():
    backend = make_backend("consent")
    try:
        assert backend.reset().alive
        strategies = join_strategies(backend)
        assert "dismiss-consent" in strategies
        assert backend.driver.execute_script("return window.consentDismissed === true")
        assert backend.driver.execute_script(
            "return document.querySelector('.fc-consent-root') === null"
        )
    finally:
        backend.close()


def test_dead_menu_raises_timeout_with_diagnosis():
    backend = make_backend("deadmenu", play_timeout=4.0)
    try:
        with pytest.raises(TimeoutError) as excinfo:
            backend.reset()
        message = str(excinfo.value)
        assert "menu diagnosis" in message
        assert backend.join_diagnosis is not None
        assert backend.join_diagnosis["globals"]["connect"] == "undefined"
        assert "nothing-applicable" in join_strategies(backend)
    finally:
        backend.close()


def test_probe_against_fake_page(tmp_path, capsys):
    from types import SimpleNamespace

    from slither_bot.live.probe import cmd_probe

    args = SimpleNamespace(
        url=page_url(),
        out=str(tmp_path),
        fixture=str(tmp_path / "live_dump.json"),
        play_timeout=20.0,
        manual_join_timeout=5.0,
        server=None,
    )
    rc = cmd_probe(args)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "probe PASSED" in out
    assert (tmp_path / "probe_report.json").exists()
    assert (tmp_path / "live_dump.json").exists()
    report = json.loads((tmp_path / "probe_report.json").read_text())
    assert report["grd_in_game"] == 21600


def test_renamed_globals_fail_fast_with_targeted_message():
    backend = make_backend("renamed", play_timeout=15.0)
    try:
        with pytest.raises(TimeoutError) as excinfo:
            backend.reset()
        message = str(excinfo.value)
        assert "RENAMED" in message
        assert "probe" in message
        assert backend.join_diagnosis is not None
        # Fail-fast: the snake_missing strikes trip long before play_timeout.
        assert backend.join_diagnosis["globals"]["playing"] is True
    finally:
        backend.close()


def test_probe_rescues_renamed_globals_end_to_end(tmp_path, capsys):
    from types import SimpleNamespace

    from slither_bot.live import js_bridge
    from slither_bot.live.probe import cmd_probe

    args = SimpleNamespace(
        url=page_url("renamed"),
        out=str(tmp_path),
        fixture=str(tmp_path / "live_dump.json"),
        play_timeout=15.0,
        manual_join_timeout=2.0,
        server=None,
    )
    rc = cmd_probe(args)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "VERIFIED renamed globals" in out
    assert "probe PASSED" in out
    assert "Traceback" not in out
    assert js_bridge.GLOBALS["snakes"] == "slithers"  # applied for the session

    report = json.loads((tmp_path / "probe_report.json").read_text())
    assert report["renamed_globals"] == {"snake": "slither", "snakes": "slithers", "foods": "fud"}
    assert (tmp_path / "live_dump.json").exists()  # measurements + fixture still ran


def test_probe_dead_menu_reports_instead_of_crashing(tmp_path, capsys):
    from types import SimpleNamespace

    from slither_bot.live.probe import cmd_probe

    args = SimpleNamespace(
        url=page_url("deadmenu"),
        out=str(tmp_path),
        fixture=str(tmp_path / "live_dump.json"),
        play_timeout=3.0,
        manual_join_timeout=2.0,
        server=None,
    )
    rc = cmd_probe(args)
    out = capsys.readouterr().out
    assert rc == 1
    assert "Traceback" not in out
    assert "auto-join FAILED" in out
    # Headless (SLITHER_BOT_HEADLESS=1 in this env) skips the manual wait;
    # a visible session would instead prompt for a manual click.
    assert ("skipping the manual-join fallback" in out) or ("click Play manually" in out)
    report = json.loads((tmp_path / "probe_report.json").read_text())
    assert report["menu_diagnosis"] is not None
    assert report["join_error"]
