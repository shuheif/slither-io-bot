import numpy as np
import pytest

from slither_bot.geometry import (
    boundary_barrier,
    cull_by_value,
    decimate_polyline,
    nearest_k_indices,
    polyline_segments,
    segment_distances,
    wrap_angle,
)


def one_segment(p, a, b):
    dist, closest, grad = segment_distances(
        np.asarray(p, float), np.asarray([a], float), np.asarray([b], float)
    )
    return dist[0], closest[0], grad[0]


class TestSegmentDistances:
    def test_interior_projection(self):
        dist, closest, grad = one_segment((1, 1), (0, 0), (2, 0))
        assert dist == pytest.approx(1.0)
        assert closest == pytest.approx([1.0, 0.0])
        assert grad == pytest.approx([0.0, 1.0])

    def test_clamp_to_first_endpoint(self):
        dist, closest, grad = one_segment((-1, 1), (0, 0), (2, 0))
        assert dist == pytest.approx(np.sqrt(2.0))
        assert closest == pytest.approx([0.0, 0.0])
        assert grad == pytest.approx(np.array([-1.0, 1.0]) / np.sqrt(2.0))

    def test_clamp_to_second_endpoint(self):
        dist, closest, _ = one_segment((3, 1), (0, 0), (2, 0))
        assert dist == pytest.approx(np.sqrt(2.0))
        assert closest == pytest.approx([2.0, 0.0])

    def test_collinear_beyond_end(self):
        dist, closest, grad = one_segment((4, 0), (0, 0), (2, 0))
        assert dist == pytest.approx(2.0)
        assert closest == pytest.approx([2.0, 0.0])
        assert grad == pytest.approx([1.0, 0.0])

    def test_degenerate_segment_is_point_distance(self):
        dist, closest, grad = one_segment((3, 4), (0, 0), (0, 0))
        assert dist == pytest.approx(5.0)
        assert closest == pytest.approx([0.0, 0.0])
        assert grad == pytest.approx([0.6, 0.8])

    def test_point_on_segment_has_unit_fallback_gradient(self):
        dist, _, grad = one_segment((1, 0), (0, 0), (2, 0))
        assert dist == pytest.approx(0.0)
        assert np.linalg.norm(grad) == pytest.approx(1.0)

    def test_gradients_are_unit_norm(self, rng):
        p = rng.uniform(-10, 10, size=2)
        a = rng.uniform(-10, 10, size=(50, 2))
        b = rng.uniform(-10, 10, size=(50, 2))
        _, _, grad = segment_distances(p, a, b)
        assert np.linalg.norm(grad, axis=1) == pytest.approx(np.ones(50))

    def test_gradient_matches_finite_differences(self, rng):
        checked = 0
        eps = 1e-6
        while checked < 100:
            p = rng.uniform(-10, 10, size=2)
            a = rng.uniform(-10, 10, size=(1, 2))
            b = rng.uniform(-10, 10, size=(1, 2))
            dist, _, grad = segment_distances(p, a, b)
            if dist[0] < 1e-3:  # skip near-degenerate points where d is non-smooth
                continue
            fd = np.zeros(2)
            for axis in range(2):
                dp = np.zeros(2)
                dp[axis] = eps
                d_plus, _, _ = segment_distances(p + dp, a, b)
                d_minus, _, _ = segment_distances(p - dp, a, b)
                fd[axis] = (d_plus[0] - d_minus[0]) / (2 * eps)
            assert grad[0] == pytest.approx(fd, abs=1e-5)
            checked += 1

    def test_vectorized_matches_scalar_loop(self, rng):
        p = rng.uniform(-5, 5, size=2)
        a = rng.uniform(-5, 5, size=(20, 2))
        b = rng.uniform(-5, 5, size=(20, 2))
        dist, closest, grad = segment_distances(p, a, b)
        for i in range(20):
            d_i, c_i, g_i = one_segment(p, a[i], b[i])
            assert dist[i] == pytest.approx(d_i)
            assert closest[i] == pytest.approx(c_i)
            assert grad[i] == pytest.approx(g_i)


class TestPolylineHelpers:
    def test_polyline_segments_chains_points(self):
        pts = np.array([[0, 0], [1, 0], [1, 1]], float)
        a, b = polyline_segments(pts)
        assert a == pytest.approx(pts[:-1])
        assert b == pytest.approx(pts[1:])

    def test_polyline_segments_short_input(self):
        a, b = polyline_segments(np.zeros((1, 2)))
        assert a.shape == (0, 2) and b.shape == (0, 2)

    def test_decimate_keeps_endpoints(self):
        pts = np.arange(22, dtype=float).reshape(11, 2)
        out = decimate_polyline(pts, stride=3)
        assert out[0] == pytest.approx(pts[0])
        assert out[-1] == pytest.approx(pts[-1])
        assert len(out) < len(pts)

    def test_decimate_stride_one_is_identity(self):
        pts = np.arange(10, dtype=float).reshape(5, 2)
        assert decimate_polyline(pts, stride=1) == pytest.approx(pts)


class TestCulling:
    def test_cull_by_value(self):
        values = np.array([5.0, 1.0, 3.0, 10.0])
        assert set(cull_by_value(values, 4.0)) == {1, 2}

    def test_nearest_k(self):
        values = np.array([5.0, 1.0, 3.0, 10.0, 0.5])
        idx = nearest_k_indices(values, 2)
        assert set(idx) == {1, 4}

    def test_nearest_k_returns_all_when_short(self):
        assert set(nearest_k_indices(np.array([2.0, 1.0]), 5)) == {0, 1}


class TestBoundaryBarrier:
    def test_positive_inside_pointing_inward(self):
        h, grad = boundary_barrier(np.array([3.0, 0.0]), np.array([0.0, 0.0]), 10.0)
        assert h == pytest.approx(7.0)
        assert grad == pytest.approx([-1.0, 0.0])

    def test_zero_on_the_wall(self):
        h, _ = boundary_barrier(np.array([0.0, 10.0]), np.array([0.0, 0.0]), 10.0)
        assert h == pytest.approx(0.0)

    def test_center_degenerate(self):
        h, grad = boundary_barrier(np.zeros(2), np.zeros(2), 10.0)
        assert h == pytest.approx(10.0)
        assert grad == pytest.approx([0.0, 0.0])


class TestWrapAngle:
    @pytest.mark.parametrize(
        "theta,expected",
        [(0.0, 0.0), (np.pi, np.pi), (-np.pi, np.pi), (3 * np.pi, np.pi), (2 * np.pi, 0.0)],
    )
    def test_wraps(self, theta, expected):
        assert wrap_angle(theta) == pytest.approx(expected)
