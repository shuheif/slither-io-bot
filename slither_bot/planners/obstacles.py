"""Enemy bodies -> culled obstacle segment set, shared by APF and CBF.

Both planners see the exact same obstacle preprocessing so the experiment
isolates the avoidance mechanism, not the bookkeeping around it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slither_bot.core import Percept
from slither_bot.geometry import (
    decimate_polyline,
    nearest_k_indices,
    polyline_segments,
    segment_distances,
)


@dataclass
class ObstacleSet:
    """Segments near our head, with current distances/gradients precomputed.

    ``grad`` is d(dist)/d(head): the unit direction that *increases* distance
    to the segment.  ``vel`` is nonzero only for head-adjacent enemy segments
    (deeper body points trace the head's old path and barely move, so treating
    them as static is more accurate than giving them the head's velocity).
    """

    a: np.ndarray  # (N,2) segment starts
    b: np.ndarray  # (N,2) segment ends
    radius: np.ndarray  # (N,) enemy body radius
    vel: np.ndarray  # (N,2) obstacle velocity
    is_head: np.ndarray  # (N,) bool: head-adjacent segment
    dist: np.ndarray  # (N,) distance from our head to the segment
    grad: np.ndarray  # (N,2) d dist / d head
    clearance: np.ndarray  # (N,) dist - r_self - radius (margin-free)

    def __len__(self) -> int:
        return len(self.dist)

    @classmethod
    def empty(cls) -> "ObstacleSet":
        z2 = np.zeros((0, 2))
        z1 = np.zeros(0)
        return cls(z2, z2, z1, z2, np.zeros(0, dtype=bool), z1, z2, z1)


def build_obstacles(
    percept: Percept,
    d_influence: float,
    k_max: int = 40,
    stride: int = 2,
    head_pts: int = 6,
) -> ObstacleSet:
    """Collect enemy body segments within ``d_influence`` clearance of our head,
    decimated by ``stride`` and capped to the nearest ``k_max``."""
    p = percept.self_snake.head
    r_self = percept.self_snake.radius

    seg_a, seg_b, seg_r, seg_v, seg_head = [], [], [], [], []
    for enemy in percept.enemies:
        pts = enemy.body
        if len(pts) == 0:
            pts = enemy.head[None, :]
        elif np.linalg.norm(pts[0] - enemy.head) > 1e-6:
            pts = np.vstack([enemy.head[None, :], pts])

        # Cheap prefilter on point distances; the slack term covers the gap
        # between consecutive body points (their spacing is a few radii).
        d_pts = np.linalg.norm(pts - p, axis=1)
        if d_pts.min() - r_self - enemy.radius > d_influence + 5.0 * enemy.radius:
            continue

        pts = decimate_polyline(pts, stride)
        if len(pts) == 1:
            a, b = pts, pts  # point obstacle as a degenerate segment
        else:
            a, b = polyline_segments(pts)
        n = len(a)
        n_head = max(1, -(-head_pts // stride))  # ceil division
        head_mask = np.arange(n) < n_head
        v_head = enemy.speed * np.array([np.cos(enemy.heading), np.sin(enemy.heading)])

        seg_a.append(a)
        seg_b.append(b)
        seg_r.append(np.full(n, enemy.radius))
        seg_v.append(np.where(head_mask[:, None], v_head[None, :], 0.0))
        seg_head.append(head_mask)

    if not seg_a:
        return ObstacleSet.empty()

    a = np.concatenate(seg_a)
    b = np.concatenate(seg_b)
    radius = np.concatenate(seg_r)
    vel = np.concatenate(seg_v)
    is_head = np.concatenate(seg_head)

    dist, _, grad = segment_distances(p, a, b)
    clearance = dist - r_self - radius

    keep = np.flatnonzero(clearance < d_influence)
    keep = keep[nearest_k_indices(clearance[keep], k_max)]

    return ObstacleSet(
        a=a[keep],
        b=b[keep],
        radius=radius[keep],
        vel=vel[keep],
        is_head=is_head[keep],
        dist=dist[keep],
        grad=grad[keep],
        clearance=clearance[keep],
    )
