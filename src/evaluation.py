"""
Evaluation utilities:
  - Umeyama (scaled Procrustes) alignment
  - 3-D mean error computation
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.triangulation import triangulate_pointset


def umeyama(X_hat: np.ndarray, X_gt: np.ndarray):
    """
    Umeyama algorithm: find (s, R, t) minimising
      sum_i || s R x_hat_i + t - x_gt_i ||^2
    Returns the aligned point cloud.
    """
    n = len(X_hat)
    mu_h = X_hat.mean(0); mu_g = X_gt.mean(0)
    Xhc = X_hat - mu_h;  Xgc = X_gt - mu_g
    var_h = np.sum(Xhc ** 2) / n
    if var_h < 1e-12:
        return X_hat.copy()
    cov = (Xgc.T @ Xhc) / n
    U, sigma, Vt = np.linalg.svd(cov)
    D = np.diag([1., 1., np.linalg.det(U) * np.linalg.det(Vt)])
    R = U @ D @ Vt
    s = np.sum(sigma * np.diag(D)) / var_h
    t = mu_g - s * R @ mu_h
    return s * (X_hat @ R.T) + t


def mean_3d_error(X_hat: np.ndarray, X_gt: np.ndarray) -> float:
    """Mean 3-D error after Umeyama alignment."""
    aligned = umeyama(X_hat, X_gt)
    return float(np.mean(np.linalg.norm(aligned - X_gt, axis=1)))


def evaluate_2view(method_names, x1, x2, R1, t1, R2, t2, X_gt,
                   K=None):
    """
    Triangulate X_gt.shape[0] points with each method,
    return dict {method: mean_3d_error}.
    K=None uses identity (normalised coords).
    """
    if K is None:
        K = np.eye(3)
    cameras  = [(R1, t1), (R2, t2)]
    pts_sets = [x1, x2]          # each (N,2)
    errors   = {}
    for name in method_names:
        X_hat = triangulate_pointset(name, cameras, pts_sets, K)
        errors[name] = mean_3d_error(X_hat, X_gt)
    return errors


def evaluate_nview(method_names, xs, Rs, ts, X_gt, K=None):
    """
    Multi-view version. xs: list of (N,2) arrays.
    Rs: (Nc,3,3), ts: (Nc,3).
    """
    if K is None:
        K = np.eye(3)
    cameras  = [(Rs[i], ts[i]) for i in range(len(Rs))]
    errors   = {}
    for name in method_names:
        X_hat = triangulate_pointset(name, cameras, xs, K)
        errors[name] = mean_3d_error(X_hat, X_gt)
    return errors
