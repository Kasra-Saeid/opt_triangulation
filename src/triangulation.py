"""
Triangulation methods: MP, IRMP, L2, L1, AngL2, AngL1
Supports 2-view and multi-view (N >= 2).

cameras   : list of (R [3x3], t [3,])   where  P_i = K [R_i | t_i]
points_2d : list of 2-D pixel coords (one array per camera)
K         : 3x3 calibration matrix  (np.eye(3) for normalised coords)
"""
import numpy as np
from scipy.optimize import minimize


# ── helpers ──────────────────────────────────────────────────────

def _camera_center(R, t):
    return -(R.T @ np.array(t, dtype=float).flatten())

def _ray(u, K, R):
    d = R.T @ np.linalg.inv(K) @ np.array([u[0], u[1], 1.0])
    return d / np.linalg.norm(d)

def _project(R, t, K, p):
    P = K @ np.hstack([R, np.array(t, float).flatten().reshape(3,1)])
    h = P @ np.append(p, 1.0)
    return h[:2] / h[2]

def _solve(cost_fn, x0, smooth):
    if smooth:
        res = minimize(cost_fn, x0, method="L-BFGS-B",
                       options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000})
    else:
        res = minimize(cost_fn, x0, method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-8,
                                "maxiter": 10000, "adaptive": True})
    return x0.copy() if np.linalg.norm(res.x - x0) > 100 else res.x


# ── MP ────────────────────────────────────────────────────────────

def midpoint(cameras, points_2d, K):
    """Closed-form: minimise sum of squared 3-D distances from lines-of-sight."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for (R, t), u in zip(cameras, points_2d):
        c = _camera_center(R, t)
        d = _ray(u, K, R)
        M = np.eye(3) - np.outer(d, d)
        A += M; b += M @ c
    return np.linalg.solve(A, b)


# ── IRMP ──────────────────────────────────────────────────────────

def irmp(cameras, points_2d, K, max_iter=10, eps=1e-8):
    """
    Iteratively Reweighted Midpoint (Yang et al. 2019).
    Weight of camera i = 1 / distance(p, line_i).
    Minimises sum of (unsquared) 3-D distances from lines-of-sight.
    """
    rays = [(_camera_center(R, t), _ray(u, K, R))
            for (R, t), u in zip(cameras, points_2d)]
    p = midpoint(cameras, points_2d, K)
    for _ in range(max_iter):
        A = np.zeros((3, 3)); b = np.zeros(3)
        for c, d in rays:
            M = np.eye(3) - np.outer(d, d)
            dist = np.linalg.norm(M @ (p - c))
            w = 1.0 / max(dist, eps)
            A += w * M; b += w * M @ c
        p_new = np.linalg.solve(A, b)
        if np.linalg.norm(p_new - p) < eps:
            break
        p = p_new
    return p


# ── L2 ────────────────────────────────────────────────────────────

def l2(cameras, points_2d, K):
    """Minimise sum of SQUARED reprojection errors — smooth → L-BFGS-B."""
    us = [np.asarray(u, float) for u in points_2d]
    x0 = midpoint(cameras, points_2d, K)
    def cost(p):
        return sum(np.sum((_project(R, t, K, p) - u)**2)
                   for (R, t), u in zip(cameras, us))
    return _solve(cost, x0, smooth=True)


# ── L1 ────────────────────────────────────────────────────────────

def l1(cameras, points_2d, K):
    """Minimise sum of ABSOLUTE reprojection errors — non-smooth → Nelder-Mead."""
    us = [np.asarray(u, float) for u in points_2d]
    x0 = midpoint(cameras, points_2d, K)
    def cost(p):
        return sum(np.sum(np.abs(_project(R, t, K, p) - u))
                   for (R, t), u in zip(cameras, us))
    return _solve(cost, x0, smooth=False)


# ── AngL2 ─────────────────────────────────────────────────────────

def angL2(cameras, points_2d, K):
    """Minimise sum of SQUARED angular errors — smooth → L-BFGS-B."""
    rays = [(_camera_center(R, t), _ray(u, K, R))
            for (R, t), u in zip(cameras, points_2d)]
    x0 = midpoint(cameras, points_2d, K)
    def cost(p):
        total = 0.0
        for c, d in rays:
            v = p - c; n = np.linalg.norm(v)
            if n < 1e-10: return 1e10
            total += np.arccos(np.clip(np.dot(v/n, d), -1, 1))**2
        return total
    return _solve(cost, x0, smooth=True)


# ── AngL1 ─────────────────────────────────────────────────────────

def angL1(cameras, points_2d, K):
    """Minimise sum of ABSOLUTE angular errors — non-smooth → Nelder-Mead."""
    rays = [(_camera_center(R, t), _ray(u, K, R))
            for (R, t), u in zip(cameras, points_2d)]
    x0 = midpoint(cameras, points_2d, K)
    def cost(p):
        total = 0.0
        for c, d in rays:
            v = p - c; n = np.linalg.norm(v)
            if n < 1e-10: return 1e10
            total += abs(np.arccos(np.clip(np.dot(v/n, d), -1, 1)))
        return total
    return _solve(cost, x0, smooth=False)


# ── Dispatcher ────────────────────────────────────────────────────

_METHODS = {"MP": midpoint, "IRMP": irmp,
            "L2": l2, "L1": l1, "AngL2": angL2, "AngL1": angL1}

def triangulate(method_name, cameras, points_2d, K):
    return _METHODS[method_name](cameras, points_2d, K)

def triangulate_pointset(method_name, cameras, pointsets_2d, K):
    N = pointsets_2d[0].shape[0]
    X = np.zeros((N, 3))
    for i in range(N):
        X[i] = triangulate(method_name, cameras, [pts[i] for pts in pointsets_2d], K)
    return X
