"""Vectorized 2-D geometry used by the APF and CBF planners.

The central primitive is the distance from the bot's head to a batch of
line segments (enemy body polylines), together with its gradient with
respect to the head position.  Using the closest-point parameterization
``t = clip((p-a)·(b-a) / |b-a|^2, 0, 1)`` the gradient is simply the unit
vector from the closest point toward ``p``, which is automatically correct
at segment endpoints (where the distance field becomes radial).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def segment_distances(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance from point ``p`` (2,) to segments ``a[i]->b[i]`` ((N,2) each).

    Returns ``(dist, closest, grad)`` with shapes (N,), (N,2), (N,2), where
    ``grad[i] = d dist_i / d p`` is a unit vector pointing from the closest
    point on segment i toward ``p``.  For the degenerate case ``p`` exactly
    on a segment, the gradient falls back to the direction from the segment
    midpoint to ``p``, then to the segment normal, then to +x.
    """
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1, 2)

    ab = b - a
    ab_sq = np.einsum("ij,ij->i", ab, ab)
    ap = p[None, :] - a
    t = np.einsum("ij,ij->i", ap, ab) / np.maximum(ab_sq, _EPS)
    t = np.clip(np.where(ab_sq > _EPS, t, 0.0), 0.0, 1.0)
    closest = a + t[:, None] * ab
    diff = p[None, :] - closest
    dist = np.linalg.norm(diff, axis=1)
    grad = diff / np.maximum(dist, _EPS)[:, None]

    degenerate = dist < _EPS
    for i in np.flatnonzero(degenerate):
        alt = p - 0.5 * (a[i] + b[i])
        alt_norm = np.linalg.norm(alt)
        if alt_norm > _EPS:
            grad[i] = alt / alt_norm
            continue
        normal = np.array([-ab[i, 1], ab[i, 0]])
        normal_norm = np.linalg.norm(normal)
        grad[i] = normal / normal_norm if normal_norm > _EPS else np.array([1.0, 0.0])

    return dist, closest, grad


def polyline_segments(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(M,2) polyline points -> segment endpoint arrays ((M-1,2), (M-1,2))."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        empty = np.zeros((0, 2))
        return empty, empty
    return points[:-1], points[1:]


def decimate_polyline(points: np.ndarray, stride: int = 2) -> np.ndarray:
    """Every ``stride``-th point, always keeping the first and last."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if stride <= 1 or len(points) <= 2:
        return points
    idx = list(range(0, len(points), stride))
    if idx[-1] != len(points) - 1:
        idx.append(len(points) - 1)
    return points[idx]


def cull_by_value(values: np.ndarray, threshold: float) -> np.ndarray:
    """Indices with ``values[i] < threshold``."""
    return np.flatnonzero(np.asarray(values) < threshold)


def nearest_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` smallest values (unordered), all if fewer."""
    values = np.asarray(values)
    if len(values) <= k:
        return np.arange(len(values))
    return np.argpartition(values, k)[:k]


def boundary_barrier(
    p: np.ndarray, center: np.ndarray, radius: float
) -> tuple[float, np.ndarray]:
    """Distance-to-wall barrier for a circular arena.

    ``h = radius - |p - center|`` (positive inside), ``grad h = -(p-center)/|p-center|``
    (points inward, the direction that increases clearance).  The caller
    subtracts its own radius/margin from ``h``.
    """
    d = np.asarray(p, dtype=float) - np.asarray(center, dtype=float)
    n = float(np.linalg.norm(d))
    if n < _EPS:
        return float(radius), np.zeros(2)
    return float(radius - n), -d / n


def wrap_angle(theta: float) -> float:
    """Wrap to (-pi, pi]."""
    wrapped = float(np.arctan2(np.sin(theta), np.cos(theta)))
    return np.pi if wrapped == -np.pi else wrapped


def wrap_angles(theta: np.ndarray) -> np.ndarray:
    """Vectorized wrap to [-pi, pi)."""
    return np.arctan2(np.sin(theta), np.cos(theta))
