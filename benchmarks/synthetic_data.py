import cv2 as cv
import os
import numpy as np
import sys

sys.path.append(os.path.abspath('src'))

from sfm import estimate_relative_rotation_translation, view_graph_optimization 


def generate_3d_points(min_bound: np.ndarray, max_bound: np.ndarray, n: int):
    return np.random.rand(n, 3) * ((max_bound - min_bound) + min_bound).T


def look_at_rotation(c):
    c = c.ravel()
    forward = -c / np.linalg.norm(c)  # camera z-axis

    up_world = np.array([0.0, 1.0, 0.0])

    # handle degeneracy (camera on up axis)
    if abs(np.dot(forward, up_world)) > 0.999:
        up_world = np.array([1.0, 0.0, 0.0])

    right = np.cross(up_world, forward)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)

    # World -> camera rotation
    R = np.stack([right, up, forward], axis=0)
    return R


def world_to_camera(X, c):
    R = look_at_rotation(c)
    x = (X - c.T) @ R.T
    x = x / x[:, 2:]
    return x


def disturb_normal_point(x: np.ndarray, K: np.ndarray) -> np.ndarray:
    K_inv = np.linalg.inv(K)
    x = x @ K.T
    x[:, :2] += np.random.randn(x.shape[0], 2)
    x = x @ K_inv.T
    return x


def synthetic_dataset_2view():
    c1 = np.array([[-7., 3., 0.]]).T
    c2 = np.array([[-10., -3., 1.]]).T
    K = np.array([
        [300., 0,    320.],
        [0,    300., 240.],
        [0,    0,    1.  ], 
    ])
    min_bound = -np.array([[1.5, 4, 3]]).T
    max_bound = -min_bound
    n_points = 20

    for _ in range(100):
        X = generate_3d_points(min_bound, max_bound, n_points)

        x1 = world_to_camera(X, c1)
        x2 = world_to_camera(X, c2)

        x1 = disturb_normal_point(x1, K)
        x2 = disturb_normal_point(x2, K)

        x1 = x1[:, :2]
        x2 = x2[:, :2]

        item = estimate_relative_rotation_translation(x1, x2)
        R1 = np.eye(3)
        R2 = item['R']
        t1 = np.zeros((3, 1))
        t2 = item['t']

        yield (x1, x2), (R1, t1), (R2, t2)


def synthetic_dataset_3view():
    c1 = np.array([[-7., 3., 0.]]).T
    c2 = np.array([[-10., -3., 1.]]).T
    c3 = np.array([[-8., 0., -2.]]).T
    K = np.array([
        [300., 0,    320.],
        [0,    300., 240.],
        [0,    0,    1.  ], 
    ])
    min_bound = -np.array([[1.5, 4, 3]]).T
    max_bound = -min_bound
    n_points = 20

    for _ in range(100):
        X = generate_3d_points(min_bound, max_bound, n_points)

        x1 = world_to_camera(X, c1)
        x2 = world_to_camera(X, c2)
        x3 = world_to_camera(X, c3)

        x1 = disturb_normal_point(x1, K)
        x2 = disturb_normal_point(x2, K)
        x3 = disturb_normal_point(x3, K)

        x1 = x1[:, :2]
        x2 = x2[:, :2]
        x3 = x3[:, :2]

        item12 = estimate_relative_rotation_translation(x1, x2)
        item13 = estimate_relative_rotation_translation(x1, x3)
        item23 = estimate_relative_rotation_translation(x2, x3)
        R_rel = {
            (0, 1): item12['R'],
            (0, 2): item13['R'],
            (1, 2): item23['R'],
        }
        t_rel = {
            (0, 1): item12['t'],
            (0, 2): item13['t'],
            (1, 2): item23['t'],
        }
        R, t = view_graph_optimization(R_rel, t_rel, 3)
        yield (x1, x2, x3), R, t


if __name__ == '__main__':
    pass
