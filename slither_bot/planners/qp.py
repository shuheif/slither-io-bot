"""Thin adapter from the textbook QP form to the ``quadprog`` solver.

Standard form used throughout the project::

    minimize    1/2 z^T P z + q^T z
    subject to  G z <= h

``quadprog.solve_qp(P', a, C, b)`` solves::

    minimize    1/2 x^T P' x - a^T x
    subject to  C^T x >= b

so we pass ``a = -q``, ``C = -G^T``, ``b = -h``.  The problem sizes here are
tiny (3 variables, tens of rows), for which quadprog's dense active-set
method (Goldfarb–Idnani) is exact and takes microseconds.

Returns ``None`` when the solver fails (inconsistent constraints or a
numerically bad problem) instead of raising, so the control loop can fall
back gracefully mid-game.
"""

from __future__ import annotations

import numpy as np
import quadprog

_RIDGE = 1e-9


def solve_qp(
    P: np.ndarray,
    q: np.ndarray,
    G: np.ndarray | None = None,
    h: np.ndarray | None = None,
) -> np.ndarray | None:
    P = np.asarray(P, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    n = P.shape[0]
    qp_P = 0.5 * (P + P.T) + _RIDGE * np.eye(n)  # symmetrize, keep strictly PD
    qp_a = -q
    try:
        if G is None or len(G) == 0:
            return quadprog.solve_qp(qp_P, qp_a)[0]
        G = np.asarray(G, dtype=np.float64).reshape(-1, n)
        h = np.asarray(h, dtype=np.float64).reshape(-1)
        return quadprog.solve_qp(qp_P, qp_a, -G.T, -h, 0)[0]
    except ValueError:
        return None
