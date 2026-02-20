"""Run experiments corresponding to Nasiri et al. (2023).

This script generates:
  - Sensitivity plots (Position/Distance/Angle) for 3 configurations.
  - Synthetic full reconstruction experiments (2-view and 3-view): tables like Table 1/2.
  - (Optional) Fountain-P11 experiment if dataset folder exists.

Requirements:
  pip install numpy scipy matplotlib pandas opencv-python

Optional:
  - Place Fountain-P11 images in ./fountain-p11/0000.png ... 0010.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.triangulation import triangulate
from src.triangulation import triangulate_pointset
from src.synthetic_data import synthetic_dataset_2view, synthetic_dataset_3view, K_SYNTH
from src.evaluation import evaluate_2view, evaluate_nview

# For Fountain-P11 (optional)
try:
    import cv2 as cv
    from src.sfm import compute_all_camera_pairs_data, compute_all_camera_n_tuples_data
    _HAS_CV = True
except Exception:
    _HAS_CV = False


# ====================================================================
# Part 4.1 Sensitivities (same setup as paper)
# ====================================================================

def getRotationLookingAt(cameraPos, target=np.array([0., 0., 0.])):
    forward = np.array(target, dtype=float) - np.array(cameraPos, dtype=float)
    forward /= np.linalg.norm(forward)
    worldUp = np.array([0., 1., 0.])
    if abs(np.dot(forward, worldUp)) > 0.99:
        worldUp = np.array([0., 0., 1.])
    right = np.cross(worldUp, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.stack([right, up, forward], axis=0)


def getRotationRightAlongX(cameraPos, target=np.array([0., 0., 0.])):
    right = np.array([1., 0., 0.])
    lookVec = np.array(target, dtype=float) - np.array(cameraPos, dtype=float)
    forward = lookVec.copy(); forward[0] = 0.
    n = np.linalg.norm(forward)
    forward = forward / n if n > 1e-10 else np.array([0., 0., 1.])
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    return np.stack([right, up, forward], axis=0)


def makePMatrix(R, c, K):
    t = -R @ np.array(c, dtype=float)
    return K @ np.hstack([R, t.reshape(3, 1)])


def addNoise(R, c, angleStd, posStd):
    c = np.array(c, dtype=float)
    noisyC = c + np.random.normal(0, posStd, 3)
    axis = np.random.randn(3); axis /= np.linalg.norm(axis)
    angle = np.random.normal(0, angleStd)
    S = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    Rn = np.eye(3) + np.sin(angle) * S + (1 - np.cos(angle)) * (S @ S)
    return Rn @ R, noisyC


def project(P, p):
    h = P @ np.append(p, 1.)
    return h[:2] / h[2]


def sampleInSphere(r=0.25):
    az = np.random.uniform(0, 2*np.pi)
    el = np.arccos(np.random.uniform(-1, 1))
    rr = r * np.random.uniform(0, 1)**(1/3)
    return np.array([rr*np.sin(el)*np.cos(az),
                     rr*np.sin(el)*np.sin(az),
                     rr*np.cos(el)])


def angleBetween(v1, v2):
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(c, -1., 1.))


def initSensitivityConfigs():
    K = K_SYNTH.copy()
    cfgs = {}

    # Config 1
    c1, c2 = np.array([-5., -1., 0.]), np.array([-5., 1., 0.])
    R1, R2 = getRotationLookingAt(c1), getRotationLookingAt(c2)
    cfgs[1] = dict(K=K, P1=makePMatrix(R1,c1,K), R1=R1, c1=c1,
                   P2=makePMatrix(R2,c2,K), R2=R2, c2=c2)

    # Config 2
    c1, c2 = np.array([-12., 0., 0.]), np.array([-2., 0., 0.])
    R1, R2 = getRotationLookingAt(c1), getRotationLookingAt(c2)
    cfgs[2] = dict(K=K, P1=makePMatrix(R1,c1,K), R1=R1, c1=c1,
                   P2=makePMatrix(R2,c2,K), R2=R2, c2=c2)

    # Config 3
    c1, c2 = np.array([-10., 2., -1.]), np.array([-5., -2., 1.])
    R1, R2 = getRotationRightAlongX(c1), getRotationRightAlongX(c2)
    cfgs[3] = dict(K=K, P1=makePMatrix(R1,c1,K), R1=R1, c1=c1,
                   P2=makePMatrix(R2,c2,K), R2=R2, c2=c2)

    return cfgs


def sensitivity_position(cfg, methods, nIters=10, noiseLevels=range(1,11)):
    out = {m: [] for m in methods}
    for nl in noiseLevels:
        posStd = 0.01 * nl
        angStd = np.deg2rad(0.1 * nl)
        errs = {m: [] for m in methods}
        for _ in range(nIters):
            p  = sampleInSphere()
            u1 = project(cfg["P1"], p)
            u2 = project(cfg["P2"], p)
            R1n, c1n = addNoise(cfg["R1"], cfg["c1"], angStd, posStd)
            R2n, c2n = addNoise(cfg["R2"], cfg["c2"], angStd, posStd)
            cams = [(R1n, -R1n @ c1n), (R2n, -R2n @ c2n)]  # store as (R,t)
            for m in methods:
                est = triangulate(m, cams, [u1, u2], cfg["K"])
                errs[m].append(np.linalg.norm(est - p))
        for m in methods:
            out[m].append(float(np.mean(errs[m])))
    return out


def sensitivity_distance(cfg, methods, nIters=10, noiseLevels=range(1,11)):
    out = {m: [] for m in methods}
    for nl in noiseLevels:
        posStd = 0.01 * nl
        angStd = np.deg2rad(0.1 * nl)
        errs = {m: [] for m in methods}
        for _ in range(nIters):
            p1, p2 = sampleInSphere(), sampleInSphere()
            u1a, u2a = project(cfg["P1"], p1), project(cfg["P2"], p1)
            u1b, u2b = project(cfg["P1"], p2), project(cfg["P2"], p2)
            R1n, c1n = addNoise(cfg["R1"], cfg["c1"], angStd, posStd)
            R2n, c2n = addNoise(cfg["R2"], cfg["c2"], angStd, posStd)
            cams = [(R1n, -R1n @ c1n), (R2n, -R2n @ c2n)]
            trueDist = np.linalg.norm(p1 - p2)
            for m in methods:
                e1 = triangulate(m, cams, [u1a, u2a], cfg["K"])
                e2 = triangulate(m, cams, [u1b, u2b], cfg["K"])
                errs[m].append(abs(trueDist - np.linalg.norm(e1 - e2)))
        for m in methods:
            out[m].append(float(np.mean(errs[m])))
    return out


def sensitivity_angle(cfg, methods, nIters=10, noiseLevels=range(1,11)):
    out = {m: [] for m in methods}
    for nl in noiseLevels:
        posStd = 0.01 * nl
        angStd = np.deg2rad(0.1 * nl)
        errs = {m: [] for m in methods}
        for _ in range(nIters):
            p1, p2, p3 = sampleInSphere(), sampleInSphere(), sampleInSphere()
            u11,u21 = project(cfg["P1"], p1), project(cfg["P2"], p1)
            u12,u22 = project(cfg["P1"], p2), project(cfg["P2"], p2)
            u13,u23 = project(cfg["P1"], p3), project(cfg["P2"], p3)
            R1n, c1n = addNoise(cfg["R1"], cfg["c1"], angStd, posStd)
            R2n, c2n = addNoise(cfg["R2"], cfg["c2"], angStd, posStd)
            cams = [(R1n, -R1n @ c1n), (R2n, -R2n @ c2n)]
            trueAng = angleBetween(p2 - p1, p3 - p1)
            for m in methods:
                e1 = triangulate(m, cams, [u11,u21], cfg["K"])
                e2 = triangulate(m, cams, [u12,u22], cfg["K"])
                e3 = triangulate(m, cams, [u13,u23], cfg["K"])
                estAng = angleBetween(e2 - e1, e3 - e1)
                errs[m].append(abs(trueAng - estAng))
        for m in methods:
            out[m].append(float(np.mean(errs[m])))
    return out


def plot_sensitivity(allCfgResults, name, noiseLevels=range(1,11)):
    styles = {"MP": "o-", "L2": "D-", "L1": "s--", "AngL2": "^-.", "AngL1": "x:"}
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    fig.suptitle(f"{name} Error Sensitivity")
    for i, cfgId in enumerate([1, 2, 3]):
        ax = axes[i]
        for m, vals in allCfgResults[cfgId].items():
            ax.plot(list(noiseLevels), vals, styles.get(m, "o-"), label=m)
        ax.set_title(f"Configuration {cfgId}")
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("Mean Error")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"figure_sensitivity_{name.lower()}.png", dpi=150)
    plt.show()


# ====================================================================
# Part 4.2 Full reconstruction on synthetic datasets
# ====================================================================

def _stats_from_runs(run_errors):
    rows = {}
    for m, vals in run_errors.items():
        arr = np.array(vals)
        rows[m] = {
            "Mean": float(np.mean(arr)),
            "Median": float(np.median(arr)),
            "Std": float(np.std(arr)),
            "Min": float(np.min(arr)),
            "Max": float(np.max(arr)),
        }
    return pd.DataFrame(rows).T


def synthetic_experiment_2view(methods, n_runs=100):
    run_errors = {m: [] for m in methods}
    for x1, x2, (R1,t1), (R2,t2), X_gt in synthetic_dataset_2view(n_runs=n_runs):
        errs = evaluate_2view(methods, x1, x2, R1, t1, R2, t2, X_gt, K=np.eye(3))
        for m in methods:
            run_errors[m].append(errs[m])
    return _stats_from_runs(run_errors)


def synthetic_experiment_3view(methods, n_runs=100):
    run_errors = {m: [] for m in methods}
    for xs, Rs, ts, X_gt in synthetic_dataset_3view(n_runs=n_runs):
        errs = evaluate_nview(methods, xs, Rs, ts, X_gt, K=np.eye(3))
        for m in methods:
            run_errors[m].append(errs[m])
    return _stats_from_runs(run_errors)


# ====================================================================
# Part 4.3 Fountain-P11 (optional)
# ====================================================================

def load_fountain_p11(folder="fountain-p11"):
    if not _HAS_CV:
        raise RuntimeError("OpenCV not available")
    images = []
    for i in range(11):
        path = os.path.join(folder, f"{i:04d}.png")
        im = cv.imread(path, cv.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(path)
        images.append(im)
    return images


def fountain_experiment(methods, n_views=2, max_images=9, repeats_per_pair=10):
    if not _HAS_CV:
        print("OpenCV not installed. Skipping Fountain-P11.")
        return None
    if not os.path.isdir("fountain-p11"):
        print("fountain-p11 folder not found. Skipping Fountain-P11.")
        return None

    imgs = load_fountain_p11("fountain-p11")[1:-1]  # remove first/last (paper)
    imgs = imgs[:max_images]

    # Intrinsics from paper
    K = np.array([[2759.48, 0, 1520.69],
                  [0, 2764.16, 1006.81],
                  [0, 0, 1]], dtype=float)

    sift = cv.SIFT.create()
    data = []
    for i, im in enumerate(imgs):
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        data.append({"id": i, "kp": kp, "des": des})

    points, R_rel, t_rel = compute_all_camera_pairs_data(data, K)

    run_errors = {m: [] for m in methods}

    if n_views == 2:
        # for each camera pair, repeat with random subsets of correspondences
        for (i, j), (x1, x2) in points.items():
            item = {"R": R_rel[(i, j)], "t": t_rel[(i, j)]}
            R1, t1 = np.eye(3), np.zeros(3)
            R2, t2 = item["R"], item["t"]

            # Without ground truth, paper evaluates via alignment to COLMAP (?)
            # Here we implement a proxy: triangulate with MP as baseline and
            # compare relative errors is not meaningful.
            # Therefore: we only report reprojection residuals as a sanity check.
            # If you have ground-truth structure for Fountain-P11, plug it here.
            idx = np.arange(len(x1))
            for _ in range(repeats_per_pair):
                if len(idx) < 50:
                    continue
                sel = np.random.choice(idx, size=min(200, len(idx)), replace=False)
                Xs = {}
                cams = [(R1, t1), (R2, t2)]
                for m in methods:
                    X_hat = triangulate_pointset(m, cams, [x1[sel], x2[sel]], np.eye(3))
                    # mean squared reprojection (in normalised coords)
                    # (proxy because no GT 3-D for this dataset in our repo)
                    e = 0.0
                    for P, u in zip([
                        np.hstack([R1, t1.reshape(3,1)]),
                        np.hstack([R2, t2.reshape(3,1)])],
                        [x1[sel], x2[sel]]):
                        for X, ui in zip(X_hat, u):
                            x = P @ np.append(X, 1.0)
                            x = x[:2] / x[2]
                            e += np.sum((x - ui) ** 2)
                    e /= (2 * len(sel))
                    run_errors[m].append(float(e))

        return _stats_from_runs(run_errors)

    if n_views == 3:
        cam3 = compute_all_camera_n_tuples_data(R_rel, t_rel, len(imgs), 3)
        for ids, item in cam3.items():
            Rs, ts = item["R"], item["t"]
            # need correspondences across 3 views; our current pipeline doesn't
            # build tracks, so we cannot reproduce paper's Table 4 exactly.
            # Placeholder: skip.
            pass
        print("3-view Fountain needs multi-view tracks (not implemented).")
        return None


# ====================================================================
# Entry
# ====================================================================

def main():
    np.random.seed(0)
    methods = ["MP", "L2", "L1", "AngL2", "AngL1"]
    noiseLevels = range(1, 11)

    # 4.1 Sensitivities
    cfgs = initSensitivityConfigs()
    for name, fn in [("Position", sensitivity_position),
                     ("Distance", sensitivity_distance),
                     ("Angle", sensitivity_angle)]:
        allRes = {}
        for cfgId in [1, 2, 3]:
            print(f"Running sensitivity {name} — config {cfgId}...")
            allRes[cfgId] = fn(cfgs[cfgId], methods, nIters=100, noiseLevels=noiseLevels)
        plot_sensitivity(allRes, name, noiseLevels)

    # 4.2 Synthetic full reconstruction
    print("Synthetic 2-view (Table 1-style):")
    table2v = synthetic_experiment_2view(methods, n_runs=100)
    print(table2v.to_string(float_format=lambda x: f"{x:.4f}"))

    print("Synthetic 3-view (Table 2-style):")
    table3v = synthetic_experiment_3view(methods, n_runs=100)
    print(table3v.to_string(float_format=lambda x: f"{x:.4f}"))

    # 4.3 Fountain-P11 (optional)
    # NOTE: full reproduction of paper Tables 3/4 requires multi-view tracks.
    f2 = fountain_experiment(methods, n_views=2)
    if f2 is not None:
        print("Fountain-P11 proxy stats (mean squared reprojection, normalised coords):")
        print(f2.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
