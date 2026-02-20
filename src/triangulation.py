"""
Triangulation methods: MP, L2, L1, AngL2, AngL1
Supports both 2-view and multi-view (N >= 2) triangulation.

Convention:
  cameras : list of (R, t) where P = K @ [R | t]
  points_2d: list of 2-D pixel coords (one array per camera)
  K        : 3x3 calibration matrix (pass np.eye(3) for normalized coords)
"""
import numpy as np
from scipy.optimize import minimize


# ── helpers ────────────────────────────────────────────────────────

def _make_P(R, t, K):
    """Build 3x4 projection matrix P = K [R | t]."""
    t = np.array(t, dtype=float).flatten()
    return K @ np.hstack([R, t.reshape(3, 1)])


def _camera_center(R, t):
    """Camera center c = -R^T t."""
    return -(R.T @ np.array(t, dtype=float).flatten())


def _ray(u, K, R):
    """Unit direction vector of the line-of-sight for pixel u."""
    d = R.T @ np.linalg.inv(K) @ np.array([u[0], u[1], 1.0])
    return d / np.linalg.norm(d)


def _project(P, p):
    h = P @ np.append(p, 1.0)
    return h[:2] / h[2]


def _solve(cost_fn, x0, smooth):
    """
    Generic solver.
    smooth=True  -> L-BFGS-B  (differentiable: L2, AngL2)
    smooth=False -> Nelder-Mead (non-differentiable: L1, AngL1)
    Falls back to x0 if the result diverges.
    """
    if smooth:
        res = minimize(cost_fn, x0, method="L-BFGS-B",
                       options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000})
    else:
        res = minimize(cost_fn, x0, method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-8,
                                "maxiter": 10000, "adaptive": True})
    if np.linalg.norm(res.x - x0) > 100:
        return x0.copy()
    return res.x


# ── Midpoint (closed-form, any number of views) ────────────────────

def midpoint(cameras, points_2d, K):
    """
    Closed-form midpoint triangulation.
    Cost: sum of squared 3-D distances from lines-of-sight.
    Solved via linear least-squares (no iterative solver).
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for (R, t), u in zip(cameras, points_2d):
        c = _camera_center(R, t)
        d = _ray(u, K, R)
        M = np.eye(3) - np.outer(d, d)   # I - d d^T
        A += M
        b += M @ c
    return np.linalg.solve(A, b)


# ── L2: minimise sum of SQUARED reprojection errors ────────────────

def l2(cameras, points_2d, K):
    """L2 optimal — smooth cost → L-BFGS-B."""
    Ps = [_make_P(R, t, K) for R, t in cameras]
    us = [np.asarray(u, float) for u in points_2d]
    x0 = midpoint(cameras, points_2d, K)

    def cost(p):
        Xh = np.append(p, 1.0)
        return sum(np.sum((_project(P, p) - u) ** 2) for P, u in zip(Ps, us))

    return _solve(cost, x0, smooth=True)


# ── L1: minimise sum of ABSOLUTE reprojection errors ───────────────

def l1(cameras, points_2d, K):
    """L1 — non-smooth cost → Nelder-Mead."""
    Ps = [_make_P(R, t, K) for R, t in cameras]
    us = [np.asarray(u, float) for u in points_2d]
    x0 = midpoint(cameras, points_2d, K)

    def cost(p):
        return sum(np.sum(np.abs(_project(P, p) - u)) for P, u in zip(Ps, us))

    return _solve(cost, x0, smooth=False)


# ── AngL2: minimise sum of SQUARED angular errors ─────────────────

def angL2(cameras, points_2d, K):
    """Angular L2 (Lee & Civera 2019) — smooth → L-BFGS-B."""
    rays   = [(_camera_center(R, t), _ray(u, K, R))
              for (R, t), u in zip(cameras, points_2d)]
    x0 = midpoint(cameras, points_2d, K)

    def cost(p):
        total = 0.0
        for c, d in rays:
            v = p - c; n = np.linalg.norm(v)
            if n < 1e-10:
                return 1e10
            total += np.arccos(np.clip(np.dot(v / n, d), -1, 1)) ** 2
        return total

    return _solve(cost, x0, smooth=True)


# ── AngL1: minimise sum of ABSOLUTE angular errors ────────────────

def angL1(cameras, points_2d, K):
    """Angular L1 (Lee & Civera 2019) — non-smooth → Nelder-Mead."""
    rays = [(_camera_center(R, t), _ray(u, K, R))
            for (R, t), u in zip(cameras, points_2d)]
    x0 = midpoint(cameras, points_2d, K)

    def cost(p):
        total = 0.0
        for c, d in rays:
            v = p - c; n = np.linalg.norm(v)
            if n < 1e-10:
                return 1e10
            total += abs(np.arccos(np.clip(np.dot(v / n, d), -1, 1)))
        return total

    return _solve(cost, x0, smooth=False)


# ── Dispatcher ────────────────────────────────────────────────────

_METHODS = {"MP": midpoint, "L2": l2, "L1": l1, "AngL2": angL2, "AngL1": angL1}


def triangulate(method_name, cameras, points_2d, K):
    """
    Triangulate a single 3-D point.

    Parameters
    ----------
    method_name : str — one of "MP", "L2", "L1", "AngL2", "AngL1"
    cameras     : list of (R [3x3], t [3,]) pairs,  P_i = K [R_i | t_i]
    points_2d   : list of 2-D pixel coords (one per camera)
    K           : 3x3 calibration matrix
    """
    return _METHODS[method_name](cameras, points_2d, K)


def triangulate_pointset(method_name, cameras, pointsets_2d, K):
    """
    Triangulate N 3-D points.

    Parameters
    ----------
    cameras      : list of (R, t) pairs (Nc cameras)
    pointsets_2d : list of Nc arrays, each (N, 2) — observations per camera
    K            : 3x3 calibration matrix

    Returns
    -------
    X_hat : (N, 3) array of estimated 3-D points
    """
    N = pointsets_2d[0].shape[0]
    X_hat = np.zeros((N, 3))
    for i in range(N):
        pts_i = [pts[i] for pts in pointsets_2d]
        X_hat[i] = triangulate(method_name, cameras, pts_i, K)
    return X_hat
