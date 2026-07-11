"""Live slither.io backend: Chrome + Selenium, perception via JS globals.

Selenium is used for two things only: executing the snippets in
``js_bridge.py`` (perception, steering, boost, respawn) and lifecycle.
The OS mouse is never touched and nothing ever clicks the canvas, so the
2022 bot's accidental-boost-on-every-move bug cannot recur by construction.

Chrome resolution order: ``SLITHER_BOT_CHROME_BINARY`` /
``SLITHER_BOT_CHROMEDRIVER`` env overrides if set, otherwise Selenium
Manager finds (and if needed downloads) a matching driver for the installed
Chrome — on a normal macOS machine no manual setup is required.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from slither_bot.core import Action, Percept
from slither_bot.live import js_bridge


def make_driver(headless: bool = False) -> webdriver.Chrome:
    options = Options()
    binary = os.environ.get("SLITHER_BOT_CHROME_BINARY")
    if binary:
        options.binary_location = binary
    if headless or os.environ.get("SLITHER_BOT_HEADLESS") == "1":
        options.add_argument("--headless=new")
    # A fresh profile avoids "user data directory is already in use" when a
    # normal Chrome is open, and keeps the bot out of the user's real profile.
    options.add_argument(
        f"--user-data-dir={tempfile.mkdtemp(prefix='slither-bot-chrome-')}"
    )
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        options.add_argument("--no-sandbox")  # required when running as root
    # slither.io's game sockets are hardcoded insecure ws:// — if Chrome
    # silently upgrades http://slither.io to https:// (default since M117),
    # every socket is blocked as mixed content and joining hangs forever.
    # Unknown feature names in the list are ignored harmlessly.
    options.add_argument(
        "--disable-features=HttpsUpgrades,HttpsFirstBalancedMode,"
        "HttpsFirstModeV2ForEngagedSites,HttpsFirstModeIncognito"
    )
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--autoplay-policy=no-user-gesture-required")
    options.add_argument("--window-size=900,700")
    # Keep the game loop running at full rate even if the window loses focus.
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-renderer-backgrounding")
    options.add_argument("--no-first-run")
    options.add_argument("--disable-infobars")
    options.add_argument("--mute-audio")

    driver_path = os.environ.get("SLITHER_BOT_CHROMEDRIVER")
    service = Service(executable_path=driver_path) if driver_path else None
    return webdriver.Chrome(options=options, service=service)


class LiveBackend:
    def __init__(
        self,
        url: str = "http://slither.io",
        hz: float = 15.0,
        nickname: str = "cbf-bot",
        allow_boost: bool = False,
        view_radius: float = 1200.0,
        headless: bool = False,
        play_timeout: float = 30.0,
        force_server: str | None = None,
    ) -> None:
        self.url = url
        self.dt = 1.0 / hz
        self.nickname = nickname
        self.allow_boost = allow_boost
        self.view_radius = view_radius
        self.headless = headless
        self.play_timeout = play_timeout
        self.force_server = force_server or os.environ.get("SLITHER_BOT_FORCE_SERVER")
        self.death_cause: str | None = None
        self.driver: webdriver.Chrome | None = None
        self.last_join_log: list[dict] = []
        self.join_diagnosis: dict | None = None
        self._t0: float | None = None
        self._next_tick = 0.0
        self._boosting = False

    # -- lifecycle ----------------------------------------------------------
    def _ensure_driver(self) -> webdriver.Chrome:
        if self.driver is None:
            self.driver = make_driver(self.headless)
            self.driver.get(self.url)
        return self.driver

    def read_percept(self) -> Percept:
        raw = self._ensure_driver().execute_script(js_bridge.READ_STATE, self.view_radius)
        t_rel = 0.0
        if raw and raw.get("playing"):
            if self._t0 is None:
                self._t0 = float(raw["t"])
            t_rel = float(raw["t"]) - self._t0
        return js_bridge.parse_state(raw, t=t_rel)

    def reset(self, seed: int | None = None) -> Percept:  # seed is meaningless live
        driver = self._ensure_driver()
        self.death_cause = None
        self._t0 = None
        self.last_join_log = []
        self.join_diagnosis = None

        if self.force_server:
            ip, _, port = self.force_server.partition(":")
            driver.execute_script(
                "if (typeof window.forceServer === 'function')"
                "{ window.forceServer(arguments[0], arguments[1]); return true; } return false;",
                ip,
                int(port or 444),
            )
            print(f"[join] pinned server {self.force_server}")

        consent = driver.execute_script(js_bridge.DISMISS_CONSENT) or {}
        consent_found = bool(consent.get("present"))
        if consent_found or consent.get("iframes"):
            print(
                f"[join] consent overlays: clicked={consent.get('clicked')} "
                f"iframes={len(consent.get('iframes', []))}"
            )
            self.last_join_log.append({"strategy": "dismiss-consent", "state": consent})

        deadline = time.monotonic() + self.play_timeout
        force_at = time.monotonic() + self.play_timeout / 2.0
        attempt = 0
        last_line = None
        percept = self.read_percept()
        while not percept.alive and time.monotonic() < deadline:
            attempt += 1
            # After half the budget with no join, break any stuck client latch
            # by escalating to a direct connect().
            force = time.monotonic() >= force_at
            result = driver.execute_script(js_bridge.PLAY, self.nickname, force)
            if isinstance(result, dict):
                self.last_join_log.append(result)
                state = result.get("state") or {}
                line = (
                    f"{result.get('strategy')} playing={state.get('playing')} "
                    f"connecting={state.get('connecting')} want_play={state.get('want_play')} "
                    f"sos={state.get('sos_len')} {state.get('protocol')}"
                )
                if line != last_line:
                    print(f"[join] attempt {attempt}: {line}")
                    last_line = line
            if consent_found and attempt % 5 == 0:
                driver.execute_script(js_bridge.DISMISS_CONSENT)
            time.sleep(1.0)
            percept = self.read_percept()

        if not percept.alive:
            try:
                self.join_diagnosis = driver.execute_script(js_bridge.DIAGNOSE_MENU)
            except Exception:
                self.join_diagnosis = None
            raise TimeoutError(self._join_failure_message())
        self._next_tick = time.monotonic() + self.dt
        return percept

    def _join_failure_message(self) -> str:
        diag = self.join_diagnosis or {}
        g = diag.get("globals", {})
        if diag.get("protocol") == "https:":
            hint = (
                "the page loaded as https, so the game's insecure ws:// sockets are blocked "
                "as mixed content (the backend passes flags against Chrome's auto-upgrade; "
                "if this persists, allow insecure content for slither.io in Chrome settings)"
            )
        elif g.get("waiting_for_sos") or g.get("sos_len") == 0:
            hint = (
                "the server list (/i33628.txt) never loaded — slither.io outage or the "
                "request is blocked on this network"
            )
        elif g.get("connecting"):
            hint = (
                "stuck connecting: the game socket never opens — servers unreachable from "
                "this network or ws:// blocked; try --server ip:port or another network"
            )
        else:
            hint = "no join strategy took effect — the menu may have changed; see the diagnosis"
        compact = json.dumps(diag, separators=(",", ":"))[:1500] if diag else "unavailable"
        return (
            f"could not join a game within {self.play_timeout:.0f}s — {hint}.\n"
            f"menu diagnosis: {compact}\n"
            f"run `python -m slither_bot probe` for full diagnostics and a manual-join fallback"
        )

    def step(self, action: Action) -> Percept:
        driver = self._ensure_driver()
        driver.execute_script(js_bridge.STEER, float(action.heading))
        boost = bool(action.boost) and self.allow_boost
        if boost != self._boosting:
            driver.execute_script(js_bridge.BOOST, 1 if boost else 0)
            self._boosting = boost

        # Pace to the control rate on the wall clock; if a round-trip ran
        # long, skip ahead instead of accumulating debt.
        now = time.monotonic()
        if now < self._next_tick:
            time.sleep(self._next_tick - now)
            self._next_tick += self.dt
        else:
            self._next_tick = now + self.dt

        percept = self.read_percept()
        if not percept.alive:
            self.death_cause = "died"  # the live game doesn't say how
            if self._boosting:
                driver.execute_script(js_bridge.BOOST, 0)
                self._boosting = False
        return percept

    def close(self) -> None:
        if self.driver is not None:
            try:
                self.driver.quit()
            finally:
                self.driver = None
