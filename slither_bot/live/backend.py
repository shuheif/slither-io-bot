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
    ) -> None:
        self.url = url
        self.dt = 1.0 / hz
        self.nickname = nickname
        self.allow_boost = allow_boost
        self.view_radius = view_radius
        self.headless = headless
        self.play_timeout = play_timeout
        self.death_cause: str | None = None
        self.driver: webdriver.Chrome | None = None
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
        deadline = time.monotonic() + self.play_timeout
        percept = self.read_percept()
        while not percept.alive and time.monotonic() < deadline:
            driver.execute_script(js_bridge.PLAY, self.nickname)
            time.sleep(1.0)
            percept = self.read_percept()
        if not percept.alive:
            raise TimeoutError(
                f"could not join a game within {self.play_timeout:.0f}s — "
                "is the menu visible? run `python -m slither_bot probe` to check the page"
            )
        self._next_tick = time.monotonic() + self.dt
        return percept

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
