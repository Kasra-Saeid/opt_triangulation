import cv2 as cv
import numpy as np
import itertools
import scipy
from typing import Tuple


def so3_exp(w: np.ndarray) -> np.ndarray:
    """
    Exp map from so(3) to SO(3).
    Returns rotation matrix from axis-angle vector (Rodrigues vector).
    """
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)

    k = w / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    R = np.eye(3) + np.sin(theta) * K \
        + (1 - np.cos(theta)) * (K @ K)
    return R


def so3_log(R: np.ndarray) -> np.ndarray:
    """
    Log map from SO(3) to so(3).
    Returns axis-angle vector (Rodrigues vector) from rotation matrix.
    """
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Small-angle case
    if theta < 1e-12:
        return np.zeros(3)

    # Near-pi case (numerically delicate)
    if np.pi - theta < 1e-5:
        # Use diagonal elements for stability
        A = (R + np.eye(3)) / 2
        axis = np.array([
            np.sqrt(max(A[0, 0], 0)),
            np.sqrt(max(A[1, 1], 0)),
            np.sqrt(max(A[2, 2], 0)),
        ])

        # Fix signs using off-diagonals
        axis[0] = np.copysign(axis[0], R[2, 1] - R[1, 2])
        axis[1] = np.copysign(axis[1], R[0, 2] - R[2, 0])
        axis[2] = np.copysign(axis[2], R[1, 0] - R[0, 1])

        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return theta * axis

    # Normal case
    w = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2 * np.sin(theta))

    return theta * w


def vgo_rot_avg_loss(W, R_rel):
    W = W.reshape((-1, 3))
    R = np.stack([so3_exp(w) for w in W])
    N = R.shape[0]

    residuals = []
    for i, j in itertools.combinations(range(N), 2):
        residual = (R_rel[(i, j)] - R[i] @ R[j].T).ravel()
        residuals.append(residual)

    loss = np.concatenate(residuals)
    return loss


def vgo_trans_avg_loss(tlmbda: np.ndarray, t_rel: dict[Tuple[int, int], np.ndarray],
                       R: np.ndarray) -> np.ndarray:
    N = R.shape[0]
    t, lmbda = np.reshape(tlmbda[:N*3], (N, 3)), np.reshape(tlmbda[N*3:], (N, N))
    pairs = itertools.combinations(range(N), 2)
    loss = np.array(0.)
    residuals = []
    for i, j in pairs:
        residual = (lmbda[i, j] * t_rel[(i, j)] - R[i].T @ (t[j] - t[i])).ravel()
        residuals.append(residual)

    loss = np.concatenate(residuals)
    return loss


def view_graph_optimization(R_rel: dict[Tuple[int, int], np.ndarray],
                            t_rel: dict[Tuple[int, int], np.ndarray], N: int) \
                            -> Tuple[np.ndarray, np.ndarray]:
    R = [np.eye(3)] + [R_rel[(0, i)] for i in range(1, N)]
    W = [so3_log(r) for r in R]
    W = np.concatenate(W)
    res = scipy.optimize.least_squares(vgo_rot_avg_loss, W, args=(R_rel,))
    W = res.x
    W = W.reshape((-1, 3))
    R = [so3_exp(w) for w in W]
    R = np.stack(R)

    lmbda = np.ones((N**2,))
    t1 = [np.zeros((3, 1))] + [t_rel[(0, i)] for i in range(1, N)]
    tlmbda = np.concatenate([np.hstack(t1).flatten(), lmbda])
    res = scipy.optimize.least_squares(vgo_trans_avg_loss, tlmbda, args=(t_rel, R))
    tlmbda = res.x
    t = tlmbda[:N*3].reshape((N, 3))

    return R, t


def normalize_points(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    K_inv = np.linalg.inv(K)
    pts = np.hstack((pts, np.ones_like(pts[:, :1])))
    pts = pts @ K_inv.T
    pts = pts[:, :2]
    return pts


def estimate_relative_rotation_translation(pts1: np.ndarray, pts2: np.ndarray):
    E, _ = cv.findEssentialMat(pts1, pts2, method=cv.FM_LMEDS)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))

    U, _, V = np.linalg.svd(E)

    W = np.array([
        [0., -1., 0.],
        [+1., 0., 0.],
        [0., +1., 0.],
    ])
    R = U @ W @ V.T
    if np.linalg.det(R) < 0:
        R = U @ W.T @ V.T

    t = U[2:].T
    P2 = np.hstack((R, t))
    x1 = pts1[0]
    x2 = pts2[0]
    A = np.vstack([
        x1[1] * P1[2] - P1[1],
        P1[0] - x1[0] * P1[2],
        x2[1] * P2[2] - P2[1],
        P2[0] - x2[0] * P2[2],
    ])
    X = scipy.linalg.solve(A, np.zeros((4, 1)))
    if X[2] < 0:
        t = -U[2:].T

    result = {
        'R': R,
        't': t,
    }
    return result


def compute_all_camera_pairs_data(image_data, K: np.ndarray) -> Tuple[
        dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
        dict[Tuple[int, int], np.ndarray],
        dict[Tuple[int, int], np.ndarray]]:
    points, R_rel, t_rel = {}, {}, {}
    data_pairs = itertools.combinations(image_data, 2)
    for item1, item2 in data_pairs:
        id1 = item1['id']
        id2 = item2['id']
        des1 = item1['des']
        des2 = item2['des']
        kp1 = item1['kp']
        kp2 = item2['kp']

        # Match keypoints
        matcher = cv.BFMatcher()
        matches = matcher.knnMatch(des1, des2, 2)
        pts1, pts2 = [], []
        for _, (m, n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        pts1 = normalize_points(pts1, K)
        pts2 = normalize_points(pts2, K)
        points[(id1, id2)] = [pts1, pts2]

        item = estimate_relative_rotation_translation(pts1, pts2)
        R_rel[(id1, id2)] = item['R']
        t_rel[(id1, id2)] = item['t']

    return points, R_rel, t_rel


def compute_all_camera_n_tuples_data(R_rel: dict[Tuple[int, int], np.ndarray],
                                     t_rel: dict[Tuple[int, int], np.ndarray],
                                     n_images: int, N: int):
    assert N > 2
    cam_tuples = {}
    id_tuples = itertools.combinations(range(n_images), N)
    for ids in id_tuples:
        id_pairs = list(itertools.combinations(ids, 2))
        tuple_id_pairs = list(itertools.combinations(range(N), 2))
        R_tuple_rel = {
            tuple_id_pair: R_rel[id_pair]
            for tuple_id_pair, id_pair in zip(tuple_id_pairs, id_pairs)
        }
        t_tuple_rel = {
            tuple_id_pair: t_rel[id_pair]
            for tuple_id_pair, id_pair in zip(tuple_id_pairs, id_pairs)
        }

        R, t = view_graph_optimization(R_tuple_rel, t_tuple_rel, N)
        cam_tuples[ids] = {
            'R': R,
            't': t,
        }

    return cam_tuples
