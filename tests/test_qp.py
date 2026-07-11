"""QP adapter tests against hand-derived KKT solutions."""

import numpy as np
import pytest

from slither_bot.planners.qp import solve_qp

I2 = np.eye(2)


def test_unconstrained_returns_minimizer():
    z = solve_qp(I2, -np.array([1.0, 2.0]))
    assert z == pytest.approx([1.0, 2.0])


def test_single_active_constraint_projects():
    # min ||u - (1,0)||^2  s.t.  u_y >= 0.5   (as -u_y <= -0.5)
    z = solve_qp(I2, -np.array([1.0, 0.0]), np.array([[0.0, -1.0]]), np.array([-0.5]))
    assert z == pytest.approx([1.0, 0.5])


def test_inactive_constraint_is_ignored():
    z = solve_qp(I2, -np.array([1.0, 0.0]), np.array([[0.0, -1.0]]), np.array([1.0]))
    assert z == pytest.approx([1.0, 0.0])


def test_two_variable_kkt_case():
    # min 1/2||u - (1,1)||^2  s.t.  u_x + u_y <= 1  ->  u = (0.5, 0.5)
    z = solve_qp(I2, -np.array([1.0, 1.0]), np.array([[1.0, 1.0]]), np.array([1.0]))
    assert z == pytest.approx([0.5, 0.5])


def test_box_constraint_binds():
    G = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    h = np.array([2.0, 2.0, 2.0, 2.0])
    z = solve_qp(I2, -np.array([10.0, 0.0]), G, h)
    assert z == pytest.approx([2.0, 0.0])


def test_slack_variable_rescues_infeasible_constraints():
    # CBF-like structure: z = [ux, uy, delta], opposing half-planes that are
    # only satisfiable with delta >= 1:
    #   ux - delta <= -1,   -ux - delta <= -1,   -delta <= 0
    # With rho >> 1 the optimum clamps to delta = 1, ux = 0 (hand-derived).
    rho = 1e3
    P = np.diag([1.0, 1.0, rho])
    q = -np.array([1.0, 0.0, 0.0])  # u_nom = (1, 0)
    G = np.array(
        [
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    h = np.array([-1.0, -1.0, 0.0])
    z = solve_qp(P, q, G, h)
    assert z is not None
    assert z[2] == pytest.approx(1.0, abs=1e-6)
    assert z[0] == pytest.approx(0.0, abs=1e-6)


def test_truly_infeasible_returns_none():
    # u_x <= -1 and u_x >= 1 with no slack: inconsistent.
    G = np.array([[1.0, 0.0], [-1.0, 0.0]])
    h = np.array([-1.0, -1.0])
    assert solve_qp(I2, np.zeros(2), G, h) is None
