"""
Synthetic dataset generators for 2-view and 3-view experiments.
Camera positions match Figure 6 / Figure 8 of the paper.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sfm import estimate_relative_rotation_translation, view_graph_optimization


K_SYNTH = np.array([[300., 0., 320.],
                     [0., 300., 240.],
                     [0., 0.,    1.]])


def _look_at(c):
    """Rotation matrix for a camera at c looking at the origin."""
    c = np.array(c, dtype=float).flatten()
    forward = -c / np.linalg.norm(c)
    up_world = np.array([0., 1., 0.])
    if abs(np.dot(forward, up_world)) > 0.999:
        up_world = np.array([1., 0., 0.])
    right = np.cross(up_world, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.stack([right, up, forward], axis=0)


def _project_normalised(X, c):
    """Project Nx3 world points to Nx2 normalised image coords."""
    R = _look_at(c)
    Xc = (X - c) @ R.T          # Nx3 camera coords
    return Xc[:, :2] / Xc[:, 2:3]   # Nx2 normalised


def _add_pixel_noise(x_norm, K, sigma=1.0):
    """Add Gaussian noise of `sigma` pixels, return normalised coords."""
    K_inv = np.linalg.inv(K)
    h = np.hstack([x_norm, np.ones((len(x_norm), 1))])
    pixel = h @ K.T
    pixel[:, :2] += np.random.randn(*pixel[:, :2].shape) * sigma
    return (pixel @ K_inv.T)[:, :2]


def _random_box_points(n=20):
    """N random points inside the 3x8x6 box centred at origin (paper Fig.6)."""
    return np.random.uniform([-1.5, -4., -3.], [1.5, 4., 3.], (n, 3))


def synthetic_dataset_2view(n_runs=100):
    """
    Yields (x1, x2, (R1,t1), (R2,t2), X_gt) for each run.
    x1, x2: (N,2) normalised image coords with 1-pixel noise.
    R1,t1 = I,0  (reference camera)
    R2,t2 from estimated essential matrix.
    X_gt: (N,3) ground truth 3-D points (world frame).
    """
    c1 = np.array([-7.,  3., 0.])
    c2 = np.array([-10., -3., 1.])
    for _ in range(n_runs):
        X_gt = _random_box_points()
        x1 = _add_pixel_noise(_project_normalised(X_gt, c1), K_SYNTH)
        x2 = _add_pixel_noise(_project_normalised(X_gt, c2), K_SYNTH)
        result = estimate_relative_rotation_translation(x1, x2)
        R2, t2 = result["R"], result["t"].flatten()
        R1, t1 = np.eye(3), np.zeros(3)
        yield x1, x2, (R1, t1), (R2, t2), X_gt


def synthetic_dataset_3view(n_runs=100):
    """
    Yields (xs, Rs, ts, X_gt) for each run.
    xs: list of 3 arrays (N,2), normalised coords with noise.
    Rs: (3,3,3) estimated rotations, ts: (3,3) estimated translations.
    X_gt: (N,3) ground truth.
    """
    c1 = np.array([-7.,  3.,  0.])
    c2 = np.array([-10., -3.,  1.])
    c3 = np.array([-8.,   0., -2.])
    for _ in range(n_runs):
        X_gt = _random_box_points()
        x1 = _add_pixel_noise(_project_normalised(X_gt, c1), K_SYNTH)
        x2 = _add_pixel_noise(_project_normalised(X_gt, c2), K_SYNTH)
        x3 = _add_pixel_noise(_project_normalised(X_gt, c3), K_SYNTH)
        r12 = estimate_relative_rotation_translation(x1, x2)
        r13 = estimate_relative_rotation_translation(x1, x3)
        r23 = estimate_relative_rotation_translation(x2, x3)
        R_rel = {(0,1): r12["R"], (0,2): r13["R"], (1,2): r23["R"]}
        t_rel = {(0,1): r12["t"].flatten(),
                 (0,2): r13["t"].flatten(),
                 (1,2): r23["t"].flatten()}
        Rs, ts = view_graph_optimization(R_rel, t_rel, 3)
        yield [x1, x2, x3], Rs, ts, X_gt
