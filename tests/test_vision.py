"""Vision fallback smoke tests on the original 2022 gameplay capture."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from slither_bot.live.vision_backend import VisionConfig, detect, percept_from_detections

SAMPLE = Path(__file__).parent.parent / "images" / "capture_sample.png"


@pytest.fixture(scope="module")
def sample_image():
    image = cv2.imread(str(SAMPLE))
    assert image is not None, f"missing test asset {SAMPLE}"
    return image


def test_detects_segments_and_pellets_on_real_capture(sample_image):
    segments, pellets = detect(sample_image)
    assert len(segments) > 0
    assert len(pellets) > 0
    height, width = sample_image.shape[:2]
    assert np.all(segments[:, [0, 2]] >= 0) and np.all(segments[:, [0, 2]] <= width)
    assert np.all(segments[:, [1, 3]] >= 0) and np.all(segments[:, [1, 3]] <= height)


def test_percept_is_valid_and_planner_runs_one_tick(sample_image):
    segments, pellets = detect(sample_image)
    percept = percept_from_detections(sample_image.shape, segments, pellets, t=0.0)

    assert percept.alive
    assert percept.self_snake.speed == 0.0  # unknown -> planners use their fallback
    assert percept.arena_center is None
    head = percept.self_snake.head
    cfg = VisionConfig()
    for enemy in percept.enemies:
        assert np.linalg.norm(enemy.body - head, axis=1).min() >= cfg.self_exclusion_px

    from slither_bot.planners.apf import APFPlanner
    from slither_bot.planners.cbf import CBFPlanner

    for planner in (CBFPlanner(), APFPlanner()):
        action = planner.plan(percept)
        assert np.isfinite(action.heading)


def test_dead_percept_when_not_playing():
    percept = percept_from_detections((1, 1), np.zeros((0, 4)), np.zeros((0, 3)), 0.0, alive=False)
    assert not percept.alive
