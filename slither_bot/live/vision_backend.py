"""Degraded pixels-only fallback backend — the 2022 pipeline, adapted.

Perception comes from a browser screenshot: Canny edges, HoughLinesP line
segments as enemy bodies (replacing the dead ``pylsd`` dependency), and
small contours as pellets.  It inherits everything else (browser lifecycle,
steering, pacing) from ``LiveBackend`` — steering still writes ``xm/ym``.

Honest limitations, by design (this is the comparison baseline for the JS
backend, and the motivation for it):

* the pixel frame is *declared* to be the world frame with the head pinned
  at the screen point the camera keeps it at (center, slightly above);
* no enemy velocities (``speed=0``), no arena boundary, no score;
* edges within ``self_exclusion_px`` of the head are dropped — otherwise the
  bot's own body reads as a lethal obstacle — which also blinds it to
  genuinely-close enemies; the hex background and UI text still leak
  detections elsewhere (see ``images/capture_sample.png``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from slither_bot.core import Percept, Snake, empty_body, empty_foods
from slither_bot.live.backend import LiveBackend


@dataclass
class VisionConfig:
    canny_lo: int = 150
    canny_hi: int = 255
    hough_threshold: int = 30
    min_line_len_px: int = 15
    max_line_gap_px: int = 5
    pellet_area_px: tuple[float, float] = (2.0, 1000.0)
    self_radius_px: float = 15.0  # assumed own collision radius
    line_pad_px: float = 6.0  # assumed enemy half-width around a detected line
    self_exclusion_px: float = 80.0  # drop detections this close to our own head
    head_offset_px: float = -25.0  # head sits slightly above the screen center


def detect(image_bgr: np.ndarray, cfg: VisionConfig | None = None):
    """BGR screenshot -> (segments (N,4) [x1,y1,x2,y2], pellets (K,3) [x,y,area])."""
    cfg = cfg or VisionConfig()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cfg.canny_lo, cfg.canny_hi)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.min_line_len_px,
        maxLineGap=cfg.max_line_gap_px,
    )
    segments = lines.reshape(-1, 4).astype(float) if lines is not None else np.zeros((0, 4))

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    pellets = []
    lo, hi = cfg.pellet_area_px
    for contour in contours:
        area = cv2.contourArea(contour)
        if lo < area < hi:
            m = cv2.moments(contour)
            if m["m00"] > 0:
                pellets.append((m["m10"] / m["m00"], m["m01"] / m["m00"], area))
    pellets = np.asarray(pellets, dtype=float) if pellets else np.zeros((0, 3))
    return segments, pellets


def percept_from_detections(
    image_shape: tuple[int, ...],
    segments: np.ndarray,
    pellets: np.ndarray,
    t: float,
    cfg: VisionConfig | None = None,
    alive: bool = True,
) -> Percept:
    """Detections in pixel coordinates -> a degraded Percept (pixel = world)."""
    cfg = cfg or VisionConfig()
    height, width = image_shape[:2]
    head = np.array([width / 2.0, height / 2.0 + cfg.head_offset_px])

    enemies: list[Snake] = []
    for x1, y1, x2, y2 in segments:
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        if min(np.linalg.norm(a - head), np.linalg.norm(b - head)) < cfg.self_exclusion_px:
            continue  # almost certainly our own body
        enemies.append(
            Snake(
                head=a,
                heading=0.0,
                speed=0.0,  # unknown from a single frame
                radius=cfg.line_pad_px,
                body=np.stack([a, b]),
            )
        )

    if len(pellets):
        keep = np.linalg.norm(pellets[:, :2] - head, axis=1) >= cfg.self_exclusion_px
        foods = np.column_stack([pellets[keep, :2], np.ones(int(keep.sum()))])
    else:
        foods = empty_foods()

    return Percept(
        t=t,
        alive=alive,
        self_snake=Snake(
            head=head, heading=0.0, speed=0.0, radius=cfg.self_radius_px, body=empty_body()
        ),
        enemies=enemies,
        foods=foods,
        arena_center=None,  # unknowable from pixels
        arena_radius=None,
        score=0.0,
    )


class VisionBackend(LiveBackend):
    """LiveBackend with perception swapped for the pixel pipeline."""

    def __init__(self, *args, vision_config: VisionConfig | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vision = vision_config or VisionConfig()
        self._episode_start = time.monotonic()

    def _is_playing(self) -> bool:
        driver = self._ensure_driver()
        try:
            playing = driver.execute_script("return !!window.playing;")
        except Exception:
            playing = None
        if playing is not None:
            return bool(playing)
        # Fallback: the menu (play button holder) is hidden while in-game.
        return bool(
            driver.execute_script(
                "const el = document.querySelector('#playh');"
                "return el ? el.offsetParent === null : false;"
            )
        )

    def read_percept(self) -> Percept:
        driver = self._ensure_driver()
        alive = self._is_playing()
        if not alive:
            return percept_from_detections((1, 1), np.zeros((0, 4)), np.zeros((0, 3)), 0.0,
                                           self.vision, alive=False)
        png = driver.get_screenshot_as_png()
        image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        segments, pellets = detect(image, self.vision)
        if self._t0 is None:
            self._t0 = time.monotonic()
        return percept_from_detections(
            image.shape, segments, pellets, time.monotonic() - self._t0, self.vision
        )
