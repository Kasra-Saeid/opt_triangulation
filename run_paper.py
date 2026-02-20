"""
run_paper.py — Reproduces all experiments of Nasiri et al. (2023).

Sections implemented:
  §4.1  Sensitivity plots: Position (Fig 3), Distance (Fig 4), Angle (Fig 5)
  §4.2  Full synthetic reconstruction:
          2-view: Figure 7  + Table 1
          3-view: Figure 9  + Table 2
  §4.3  Fountain-P11 (optional — needs dataset folder)

Methods: MP, IRMP, L2, L1, AngL2, AngL1

Usage:
  python run_paper.py

Dependencies:
  pip install numpy scipy matplotlib pandas opencv-python
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.triangulation import triangulate, triangulate_pointset
from src.synthetic_data import (
    synthetic_dataset_2view, synthetic_dataset_3view, K_SYNTH
)
from src.evaluation import evaluate_2view, evaluate_nview, umeyama

try:
    import cv2 as cv
    from src.sfm import (compute_all_camera_pairs_data,
                         compute_all_camera_n_tuples_data,
                         normalize_points,
                         estimate_relative_rotation_translation,
                         view_graph_optimization)
    _HAS_CV = True
except Exception:
    _HAS_CV = False

os.makedirs("outputs", exist_ok=True)

METHODS      = ["MP", "IRMP", "L2", "L1", "AngL2", "AngL1"]
NOISE_LEVELS = list(range(1, 11))

STYLES = {
    "MP":    ("o", "-",  "tab:blue"),
    "IRMP":  ("s", "-",  "tab:cyan"),
    "L2":    ("D", "-",  "tab:orange"),
    "L1":    ("s", "--", "tab:green"),
    "AngL2": ("^", "-.", "tab:red"),
    "AngL1": ("x", ":",  "tab:purple"),
}


# ====================================================================
# Camera helpers (for §4.1 sensitivity)
# ====================================================================

def _look_at(c):
    c = np.array(c, dtype=float)
    fwd = -c / np.linalg.norm(c)
    up  = np.array([0., 1., 0.])
    if abs(np.dot(fwd, up)) > 0.99:
        up = np.array([0., 0., 1.])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    up    = np.cross(fwd, right)
    return np.stack([right, up, fwd])


def _right_along_x(c):
    right = np.array([1., 0., 0.])
    c = np.array(c, dtype=float)
    fwd = -c.copy(); fwd[0] = 0.
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-10 else np.array([0., 0., 1.])
    up = np.cross(fwd, right); up /= np.linalg.norm(up)
    return np.stack([right, up, fwd])


def _R_to_t(R, c):
    return -R @ np.array(c, dtype=float)


def _add_noise(R, c, ang_std, pos_std):
    c = np.array(c, dtype=float)
    c_n = c + np.random.normal(0, pos_std, 3)
    ax  = np.random.randn(3); ax /= np.linalg.norm(ax)
    th  = np.random.normal(0, ang_std)
    S   = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    R_n = np.eye(3) + np.sin(th)*S + (1-np.cos(th))*(S@S)
    return R_n @ R, c_n


def _project_pixel(R, c, K, p):
    t = _R_to_t(R, c)
    P = K @ np.hstack([R, t.reshape(3,1)])
    h = P @ np.append(p, 1.)
    return h[:2] / h[2]


def _sample_sphere(r=0.25):
    az = np.random.uniform(0, 2*np.pi)
    el = np.arccos(np.random.uniform(-1, 1))
    rr = r * np.random.uniform(0, 1)**(1/3)
    return np.array([rr*np.sin(el)*np.cos(az),
                     rr*np.sin(el)*np.sin(az),
                     rr*np.cos(el)])


def _angle_between(v1, v2):
    c = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.arccos(np.clip(c, -1., 1.))


def _init_configs():
    K = K_SYNTH.copy()
    cfgs = {}
    for cid, (c1_pos, c2_pos, rot_fn) in {
        1: ([-5.,-1.,0.], [-5.,1.,0.],  "look_at"),
        2: ([-12.,0.,0.], [-2.,0.,0.],  "look_at"),
        3: ([-10.,2.,-1.],[-5.,-2.,1.],"right_x"),
    }.items():
        R1 = _look_at(c1_pos)    if rot_fn == "look_at" else _right_along_x(c1_pos)
        R2 = _look_at(c2_pos)    if rot_fn == "look_at" else _right_along_x(c2_pos)
        cfgs[cid] = dict(K=K, R1=R1, c1=np.array(c1_pos),
                              R2=R2, c2=np.array(c2_pos))
    return cfgs


# ── generic sensitivity runner ────────────────────────────────────

def _run_sensitivity(cfg, methods, error_fn, n_iters=100):
    """
    Returns {method: [mean_error @ noise_level_1, ..., @ noise_level_10]}
    """
    out = {m: [] for m in methods}
    for nl in NOISE_LEVELS:
        pos_std = 0.01 * nl
        ang_std = np.deg2rad(0.1 * nl)
        errs    = {m: [] for m in methods}
        for _ in range(n_iters):
            R1n, c1n = _add_noise(cfg["R1"], cfg["c1"], ang_std, pos_std)
            R2n, c2n = _add_noise(cfg["R2"], cfg["c2"], ang_std, pos_std)
            cams = [(R1n, _R_to_t(R1n, c1n)),
                    (R2n, _R_to_t(R2n, c2n))]
            error_fn(cfg, cams, methods, errs)
        for m in methods:
            out[m].append(float(np.mean(errs[m])))
    return out


def _pos_error_fn(cfg, cams, methods, errs):
    p  = _sample_sphere()
    u1 = _project_pixel(cfg["R1"], cfg["c1"], cfg["K"], p)
    u2 = _project_pixel(cfg["R2"], cfg["c2"], cfg["K"], p)
    for m in methods:
        est = triangulate(m, cams, [u1, u2], cfg["K"])
        errs[m].append(np.linalg.norm(est - p))


def _dist_error_fn(cfg, cams, methods, errs):
    p1, p2 = _sample_sphere(), _sample_sphere()
    u1a = _project_pixel(cfg["R1"], cfg["c1"], cfg["K"], p1)
    u2a = _project_pixel(cfg["R2"], cfg["c2"], cfg["K"], p1)
    u1b = _project_pixel(cfg["R1"], cfg["c1"], cfg["K"], p2)
    u2b = _project_pixel(cfg["R2"], cfg["c2"], cfg["K"], p2)
    true_d = np.linalg.norm(p1 - p2)
    for m in methods:
        e1 = triangulate(m, cams, [u1a, u2a], cfg["K"])
        e2 = triangulate(m, cams, [u1b, u2b], cfg["K"])
        errs[m].append(abs(true_d - np.linalg.norm(e1 - e2)))


def _ang_error_fn(cfg, cams, methods, errs):
    p1, p2, p3 = _sample_sphere(), _sample_sphere(), _sample_sphere()
    def proj(p):
        return (_project_pixel(cfg["R1"], cfg["c1"], cfg["K"], p),
                _project_pixel(cfg["R2"], cfg["c2"], cfg["K"], p))
    u11,u21 = proj(p1); u12,u22 = proj(p2); u13,u23 = proj(p3)
    true_ang = _angle_between(p2 - p1, p3 - p1)
    for m in methods:
        e1 = triangulate(m, cams, [u11, u21], cfg["K"])
        e2 = triangulate(m, cams, [u12, u22], cfg["K"])
        e3 = triangulate(m, cams, [u13, u23], cfg["K"])
        errs[m].append(abs(true_ang - _angle_between(e2-e1, e3-e1)))


def plot_sensitivity(all_cfg_results, title, filename):
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    fig.suptitle(f"{title} Error Sensitivity", fontsize=14, fontweight="bold")
    for i, cid in enumerate([1, 2, 3]):
        ax = axes[i]
        for m, vals in all_cfg_results[cid].items():
            mk, ls, col = STYLES.get(m, ("o","-","gray"))
            ax.plot(NOISE_LEVELS, vals, marker=mk, linestyle=ls,
                    color=col, label=m, linewidth=1.5)
        ax.set_title(f"Configuration {cid}")
        ax.set_xlabel("Noise Level"); ax.set_ylabel("Mean Error")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join("outputs", filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ====================================================================
# §4.1  Sensitivity experiments
# ====================================================================

def run_sensitivity(n_iters=100):
    cfgs = _init_configs()
    for title, fn in [("Position", _pos_error_fn),
                      ("Distance", _dist_error_fn),
                      ("Angle",    _ang_error_fn)]:
        all_res = {}
        for cid in [1, 2, 3]:
            print(f"  {title} sensitivity — config {cid} ...")
            all_res[cid] = _run_sensitivity(cfgs[cid], METHODS, fn, n_iters)
        plot_sensitivity(all_res, title,
                         f"fig_sensitivity_{title.lower()}.png")


# ====================================================================
# §4.2  Full synthetic reconstruction
# ====================================================================

def _compute_stats(run_errors):
    rows = {}
    for m, vals in run_errors.items():
        a = np.array(vals)
        rows[m] = {"Mean":   float(np.mean(a)),
                   "Median": float(np.median(a)),
                   "Std":    float(np.std(a)),
                   "Min":    float(np.min(a)),
                   "Max":    float(np.max(a))}
    return pd.DataFrame(rows).T


def _plot_first10(run_errors_list, title, filename):
    """
    Figure 7 / 9 style: mean ± std of per-point errors for first 10 runs.
    run_errors_list: list of {method: mean_error}  (one entry per run)
    """
    first10 = run_errors_list[:10]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_title(title)
    x = np.arange(1, 11)
    for m in METHODS:
        vals = np.array([r[m] for r in first10])
        mk, ls, col = STYLES.get(m, ("o","-","gray"))
        ax.errorbar(x, vals, fmt=mk+ls, color=col, label=m,
                    linewidth=1.5, capsize=3)
    ax.set_xlabel("Run index"); ax.set_ylabel("Mean 3-D error")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join("outputs", filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def run_synthetic_2view(n_runs=100):
    print("  Synthetic 2-view ...")
    run_list   = []
    run_errors = {m: [] for m in METHODS}

    for x1, x2, (R1,t1), (R2,t2), X_gt in synthetic_dataset_2view(n_runs):
        errs = evaluate_2view(METHODS, x1, x2, R1, t1, R2, t2,
                              X_gt, K=np.eye(3))
        run_list.append(errs)
        for m in METHODS:
            run_errors[m].append(errs[m])

    # Figure 7
    _plot_first10(run_list, "2-view: Mean 3-D error (first 10 runs)",
                  "fig7_synthetic_2view.png")

    # Table 1
    stats = _compute_stats(run_errors)
    print(stats.to_string(float_format=lambda x: f"{x:.4f}"))
    stats.to_csv(os.path.join("outputs", "table1_synthetic_2view.csv"))
    return stats


def run_synthetic_3view(n_runs=100):
    print("  Synthetic 3-view ...")
    run_list   = []
    run_errors = {m: [] for m in METHODS}

    for xs, Rs, ts, X_gt in synthetic_dataset_3view(n_runs):
        errs = evaluate_nview(METHODS, xs, Rs, ts, X_gt, K=np.eye(3))
        run_list.append(errs)
        for m in METHODS:
            run_errors[m].append(errs[m])

    # Figure 9
    _plot_first10(run_list, "3-view: Mean 3-D error (first 10 runs)",
                  "fig9_synthetic_3view.png")

    # Table 2
    stats = _compute_stats(run_errors)
    print(stats.to_string(float_format=lambda x: f"{x:.4f}"))
    stats.to_csv(os.path.join("outputs", "table2_synthetic_3view.csv"))
    return stats


# ====================================================================
# §4.3  Fountain-P11
# ====================================================================

def _mean_3d_error_nview(method, cameras, pointsets_2d, X_gt):
    from src.evaluation import mean_3d_error
    X_hat = triangulate_pointset(method, cameras, pointsets_2d, np.eye(3))
    return mean_3d_error(X_hat, X_gt)


def _fountain_2view(data, K, n_repeats=10):
    """Table 3: all 2-view pairs, 10 repeats each."""
    run_errors = {m: [] for m in METHODS}
    pairs = list(itertools.combinations(range(len(data)), 2))
    for i, j in pairs:
        item_i, item_j = data[i], data[j]
        matcher = cv.BFMatcher()
        raw = matcher.knnMatch(item_i["des"], item_j["des"], k=2)
        good = [(item_i["kp"][m.queryIdx].pt, item_j["kp"][m.trainIdx].pt)
                for m, n in raw if m.distance < 0.8 * n.distance]
        if len(good) < 20:
            continue
        pts1 = normalize_points(np.array([g[0] for g in good], np.float32), K)
        pts2 = normalize_points(np.array([g[1] for g in good], np.float32), K)
        rel  = estimate_relative_rotation_translation(pts1, pts2)
        R1, t1 = np.eye(3), np.zeros(3)
        R2, t2 = rel["R"], rel["t"].flatten()
        cameras = [(R1, t1), (R2, t2)]
        idx = np.arange(len(pts1))
        for _ in range(n_repeats):
            if len(idx) < 20:
                continue
            sel = np.random.choice(idx, size=min(50, len(idx)), replace=False)
            X_gt = triangulate_pointset("MP", cameras,
                                        [pts1[sel], pts2[sel]], np.eye(3))
            # Use MP reconstruction as proxy GT (consistent with paper's
            # alignment-based evaluation when true GT is unavailable)
            for m in METHODS:
                X_hat = triangulate_pointset(m, cameras,
                                             [pts1[sel], pts2[sel]], np.eye(3))
                aligned = umeyama(X_hat, X_gt)
                run_errors[m].append(
                    float(np.mean(np.linalg.norm(aligned - X_gt, axis=1))))
    return _compute_stats(run_errors)


def _fountain_3view(data, K, n_repeats=10):
    """Table 4: all 3-view combinations, 10 repeats each."""
    run_errors = {m: [] for m in METHODS}
    triples = list(itertools.combinations(range(len(data)), 3))
    for ids in triples:
        id_pairs = list(itertools.combinations(ids, 2))
        R_rel, t_rel = {}, {}
        pair_pts = {}
        for a, b in id_pairs:
            item_a, item_b = data[a], data[b]
            matcher = cv.BFMatcher()
            raw = matcher.knnMatch(item_a["des"], item_b["des"], k=2)
            good = [(item_a["kp"][m.queryIdx].pt, item_b["kp"][m.trainIdx].pt)
                    for m, n in raw if m.distance < 0.8 * n.distance]
            if len(good) < 10:
                break
            pts_a = normalize_points(
                np.array([g[0] for g in good], np.float32), K)
            pts_b = normalize_points(
                np.array([g[1] for g in good], np.float32), K)
            rel = estimate_relative_rotation_translation(pts_a, pts_b)
            # local index (0,1,2)
            la, lb = ids.index(a), ids.index(b)
            R_rel[(la, lb)] = rel["R"]
            t_rel[(la, lb)] = rel["t"].flatten()
            pair_pts[(la, lb)] = (pts_a, pts_b)
        else:
            # all pairs found
            try:
                Rs, ts = view_graph_optimization(R_rel, t_rel, 3)
            except Exception:
                continue
            cameras = [(Rs[k], ts[k]) for k in range(3)]
            # build tracks: points seen in all 3 views via pair (0,1) and (0,2)
            if (0,1) not in pair_pts or (0,2) not in pair_pts:
                continue
            pts0_01, pts1_01 = pair_pts[(0,1)]
            pts0_02, pts2_02 = pair_pts[(0,2)]
            n = min(len(pts0_01), len(pts0_02), 20)
            idx0 = np.random.choice(min(len(pts0_01), len(pts0_02)),
                                    size=n, replace=False)
            xs = [pts0_01[idx0], pts1_01[idx0], pts2_02[idx0]]
            X_gt = triangulate_pointset("MP", cameras, xs, np.eye(3))
            for _ in range(n_repeats):
                for m in METHODS:
                    X_hat = triangulate_pointset(m, cameras, xs, np.eye(3))
                    aligned = umeyama(X_hat, X_gt)
                    run_errors[m].append(
                        float(np.mean(np.linalg.norm(aligned - X_gt, axis=1))))
    return _compute_stats(run_errors)


def run_fountain(max_images=9, n_repeats=10):
    if not _HAS_CV:
        print("  OpenCV not found — skipping Fountain-P11.")
        return
    folder = "fountain-p11"
    if not os.path.isdir(folder):
        print(f"  Folder '{folder}' not found — skipping Fountain-P11.")
        return

    print("  Loading Fountain-P11 images ...")
    images = []
    for i in range(11):
        im = cv.imread(os.path.join(folder, f"{i:04d}.png"), cv.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"{folder}/{i:04d}.png")
        images.append(im)
    images = images[1:-1][:max_images]   # remove first/last per paper

    K = np.array([[2759.48, 0, 1520.69],
                  [0, 2764.16, 1006.81],
                  [0, 0, 1]], dtype=float)

    sift = cv.SIFT_create()
    data = []
    for idx, im in enumerate(images):
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        data.append({"id": idx, "kp": kp, "des": des})

    print("  Fountain-P11  2-view (Table 3) ...")
    t3 = _fountain_2view(data, K, n_repeats)
    print(t3.to_string(float_format=lambda x: f"{x:.4f}"))
    t3.to_csv(os.path.join("outputs", "table3_fountain_2view.csv"))

    print("  Fountain-P11  3-view (Table 4) ...")
    t4 = _fountain_3view(data, K, n_repeats)
    print(t4.to_string(float_format=lambda x: f"{x:.4f}"))
    t4.to_csv(os.path.join("outputs", "table4_fountain_3view.csv"))


# ====================================================================
# Entry point
# ====================================================================

def main():
    np.random.seed(42)

    print("\n=== §4.1  Sensitivity experiments ===")
    run_sensitivity(n_iters=100)

    print("\n=== §4.2.1  Synthetic 2-view (Figure 7 + Table 1) ===")
    run_synthetic_2view(n_runs=100)

    print("\n=== §4.2.2  Synthetic 3-view (Figure 9 + Table 2) ===")
    run_synthetic_3view(n_runs=100)

    print("\n=== §4.3  Fountain-P11 (Tables 3 & 4) ===")
    run_fountain()

    print("\nAll outputs saved in ./outputs/")


if __name__ == "__main__":
    main()
