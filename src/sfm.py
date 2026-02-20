"""
SfM pipeline utilities.
Fixes vs. original teammate code:
  1. estimate_relative_rotation_translation:
       - t = U[:, 2] (last COLUMN) — was last row (bug)
       - chirality check uses SVD-based DLT, not scipy.linalg.solve(A, 0)
       - checks all 4 E-decomposition solutions
  2. compute_all_camera_pairs_data: pts converted to float32 for cv2
"""
import cv2 as cv
import numpy as np
import itertools
import scipy.optimize
from typing import Tuple


# ── SO(3) Lie algebra utilities ───────────────────────────────────

def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues: axis-angle vector -> rotation matrix."""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector."""
    cos_t = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-12:
        return np.zeros(3)
    if np.pi - theta < 1e-5:
        A = (R + np.eye(3)) / 2
        axis = np.array([np.sqrt(max(A[i, i], 0)) for i in range(3)])
        axis[0] = np.copysign(axis[0], R[2, 1] - R[1, 2])
        axis[1] = np.copysign(axis[1], R[0, 2] - R[2, 0])
        axis[2] = np.copysign(axis[2], R[1, 0] - R[0, 1])
        return theta * axis / (np.linalg.norm(axis) + 1e-12)
    w = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2 * np.sin(theta))
    return theta * w


# ── View-graph optimisation ───────────────────────────────────────

def _vgo_rot_loss(W, R_rel):
    W = W.reshape(-1, 3)
    R = np.stack([so3_exp(w) for w in W])
    residuals = []
    for i, j in itertools.combinations(range(len(R)), 2):
        residuals.append((R_rel[(i, j)] - R[i] @ R[j].T).ravel())
    return np.concatenate(residuals)


def _vgo_trans_loss(tlmbda, t_rel, R):
    N = R.shape[0]
    t = tlmbda[:N*3].reshape(N, 3)
    lmbda = tlmbda[N*3:].reshape(N, N)
    residuals = []
    for i, j in itertools.combinations(range(N), 2):
        residuals.append(
            (lmbda[i, j] * t_rel[(i, j)] - R[i].T @ (t[j] - t[i])).ravel())
    return np.concatenate(residuals)


def view_graph_optimization(R_rel, t_rel, N):
    """Solve the viewing graph for N cameras. Returns R (N,3,3), t (N,3)."""
    W0 = np.concatenate(
        [so3_log(np.eye(3))] + [so3_log(R_rel[(0, i)]) for i in range(1, N)])
    res = scipy.optimize.least_squares(_vgo_rot_loss, W0, args=(R_rel,))
    W = res.x.reshape(-1, 3)
    R = np.stack([so3_exp(w) for w in W])

    t0_list = [np.zeros(3)] + [t_rel[(0, i)].flatten() for i in range(1, N)]
    lmbda0  = np.ones(N * N)
    x0      = np.concatenate([np.array(t0_list).flatten(), lmbda0])
    res = scipy.optimize.least_squares(_vgo_trans_loss, x0, args=(t_rel, R))
    t = res.x[:N*3].reshape(N, 3)
    return R, t


# ── Point normalisation ───────────────────────────────────────────

def normalize_points(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert pixel coords (N,2) to normalised (divided by K)."""
    K_inv = np.linalg.inv(K)
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (h @ K_inv.T)[:, :2]


# ── Relative pose estimation ──────────────────────────────────────

def _dlt_triangulate_single(P1, P2, u1, u2):
    """DLT triangulation of one point (normalised coords, K=I)."""
    A = np.array([
        u1[0] * P1[2] - P1[0],
        u1[1] * P1[2] - P1[1],
        u2[0] * P2[2] - P2[0],
        u2[1] * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def estimate_relative_rotation_translation(
        pts1: np.ndarray, pts2: np.ndarray
) -> dict:
    """
    Estimate relative R, t from normalised 2-D correspondences.
    Returns {'R': (3,3), 't': (3,1)}.
    """
    pts1f = pts1.astype(np.float32)
    pts2f = pts2.astype(np.float32)

    E, _ = cv.findEssentialMat(
        pts1f, pts2f, focal=1.0, pp=(0.0, 0.0),
        method=cv.RANSAC, prob=0.999, threshold=1e-3)

    if E is None or E.shape != (3, 3):
        return {"R": np.eye(3), "t": np.zeros((3, 1))}

    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    # FIX: t = last COLUMN of U (not last row)
    t_dir = U[:, 2]

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    best_R, best_t, best_count = None, None, -1

    for R_cand in [U @ W @ Vt, U @ W.T @ Vt]:
        if np.linalg.det(R_cand) < 0:
            R_cand = -R_cand
        for sign in [1, -1]:
            t_cand = (sign * t_dir).reshape(3, 1)
            P2 = np.hstack([R_cand, t_cand])
            # Count points in front of both cameras
            count = 0
            for u1, u2 in zip(pts1[:20], pts2[:20]):
                try:
                    X = _dlt_triangulate_single(P1, P2, u1, u2)
                    X2 = R_cand @ X + t_cand.flatten()
                    if X[2] > 0 and X2[2] > 0:
                        count += 1
                except Exception:
                    pass
            if count > best_count:
                best_count, best_R, best_t = count, R_cand, t_cand

    return {"R": best_R, "t": best_t}


# ── Batch helpers (for Fountain-P11) ─────────────────────────────

def compute_all_camera_pairs_data(image_data, K):
    points, R_rel, t_rel = {}, {}, {}
    for item1, item2 in itertools.combinations(image_data, 2):
        id1, id2 = item1["id"], item2["id"]
        matcher = cv.BFMatcher()
        matches = matcher.knnMatch(item1["des"], item2["des"], k=2)
        pts1, pts2 = [], []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                pts1.append(item1["kp"][m.queryIdx].pt)
                pts2.append(item2["kp"][m.trainIdx].pt)
        if len(pts1) < 8:
            continue
        pts1 = normalize_points(np.array(pts1, np.float32), K)
        pts2 = normalize_points(np.array(pts2, np.float32), K)
        points[(id1, id2)] = (pts1, pts2)
        item = estimate_relative_rotation_translation(pts1, pts2)
        R_rel[(id1, id2)] = item["R"]
        t_rel[(id1, id2)] = item["t"].flatten()
    return points, R_rel, t_rel


def compute_all_camera_n_tuples_data(R_rel, t_rel, n_images, N):
    assert N > 2
    cam_tuples = {}
    for ids in itertools.combinations(range(n_images), N):
        id_pairs   = list(itertools.combinations(ids, 2))
        local_pairs = list(itertools.combinations(range(N), 2))
        R_sub = {lp: R_rel[ip] for lp, ip in zip(local_pairs, id_pairs)
                 if ip in R_rel}
        t_sub = {lp: t_rel[ip] for lp, ip in zip(local_pairs, id_pairs)
                 if ip in t_rel}
        if len(R_sub) < len(local_pairs):
            continue
        R, t = view_graph_optimization(R_sub, t_sub, N)
        cam_tuples[ids] = {"R": R, "t": t}
    return cam_tuples
