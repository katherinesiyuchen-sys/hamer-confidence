"""Evaluation: does predicted confidence track actual pose error?

Three standard views of that question, plus the downstream filtering
experiment that connects the project to robot-learning-from-video.

  1. correlation: Pearson/Spearman between score and error
  2. sparsification: remove most-uncertain frames first, watch error drop
  3. risk thresholding: AUROC on "is this frame's error > tau?"
  4. filtering: keep frames with score < threshold; how much cleaner is the surviving demonstration data?
"""

from __future__ import annotations
import numpy as np
from scipy import stats

# Pose error

def mpjpe(pred: np.ndarray, gt: np.ndarray, align_root: bool = True) -> np.ndarray:
    """Mean per-joint position error, per frame, in the units of the input.

    pred, gt: (N, 21, 3). Root-aligns (joint 0 = wrist) by default, the
    standard protocol so global translation error doesn't dominate.
    Returns (N,) per-frame errors.
    """
    if align_root:
        pred = pred - pred[:, :1]
        gt = gt - gt[:, :1]
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Procrustes-aligned MPJPE: best similarity transform per frame first.

    Removes global rotation/scale so only articulation error remains —
    report both; the gap between them is itself informative.
    """
    out = np.zeros(len(pred))
    for i, (p, g) in enumerate(zip(pred, gt)):
        p = p - p.mean(0)
        g = g - g.mean(0)
        H = p.T @ g
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        scale = (S * np.diag(D)).sum() / (p ** 2).sum()
        out[i] = np.linalg.norm(scale * p @ R.T - g, axis=-1).mean()
    return out


# 1. Correlation

def score_error_correlation(scores: np.ndarray, errors: np.ndarray) -> dict:
    """Pearson (linear) and Spearman (rank) correlation between confidence
    scores and actual errors. Spearman is the headline number: we care about
    *ranking* frames correctly more than a linear fit.
    Reference point: the 2025 aleatoric-uncertainty paper reports rho ~= 0.6.
    """
    keep = ~(np.isnan(scores) | np.isnan(errors))
    s, e = scores[keep], errors[keep]
    return {
        "pearson_r": float(stats.pearsonr(s, e)[0]),
        "spearman_rho": float(stats.spearmanr(s, e)[0]),
        "n": int(keep.sum()),
    }


# 2. Sparsification

def sparsification_curve(scores: np.ndarray, errors: np.ndarray, n_points: int = 20):
    """Remove the highest-score (most uncertain) fraction of frames and track
    the mean error of what remains.

    Returns (fractions_removed, mean_error_remaining, oracle_curve, ause):
      - oracle removes by TRUE error (best possible).
      - AUSE = area between the two curves; lower = better uncertainty.
    A useless (random) score gives a flat curve; a perfect one hugs the oracle.
    """
    order = np.argsort(-scores)    # most uncertain first
    oracle_order = np.argsort(-errors)
    fracs = np.linspace(0.0, 0.99, n_points)
    curve, oracle = np.zeros(n_points), np.zeros(n_points)
    N = len(errors)
    for i, f in enumerate(fracs):
        k = int(f * N)
        curve[i] = errors[order[k:]].mean()
        oracle[i] = errors[oracle_order[k:]].mean()
    base = errors.mean()
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x / 1.x
    ause = float(trapezoid(curve - oracle, fracs) / base)   # normalized
    return fracs, curve, oracle, ause


# 3. Risk thresholding

def failure_auroc(scores: np.ndarray, errors: np.ndarray, error_thresh_mm: float = 20.0) -> float:
    """AUROC for detecting failure frames (error > threshold) from the score.

    Threshold ~20mm root-relative is a reasonable 'grasp-breaking' error for
    hand-size objects — but report a sweep (10/20/30mm), don't cherry-pick.
    Rank-based AUROC implementation (no sklearn dependency).
    """
    labels = (errors > error_thresh_mm).astype(int)
    n_pos, n_neg = labels.sum(), (1 - labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


# 4. Downstream filtering experiment

def filtering_experiment(
    scores: np.ndarray,
    errors: np.ndarray,
    keep_fractions: tuple[float, ...] = (1.0, 0.9, 0.75, 0.5, 0.25),
) -> list[dict]:
    """The robot-learning payoff: if a demonstration pipeline kept only the
    most-confident fraction of frames, how much cleaner is the kept data?

    Reports, per keep-fraction: mean/median error of kept frames and the
    fraction of failure frames (>20mm) that survived the filter. This is the
    quantitative version of 'treat reconstruction as an unreliable reference'.
    """
    order = np.argsort(scores)  # most confident first
    N = len(scores)
    rows = []
    for kf in keep_fractions:
        kept = order[: int(kf * N)]
        rows.append({
            "keep_fraction": kf,
            "mean_error_mm": float(errors[kept].mean()),
            "median_error_mm": float(np.median(errors[kept])),
            "failure_rate": float((errors[kept] > 20.0).mean()),
        })
    return rows


# Stratified reporting

def stratify_by_bin(values: np.ndarray, bins: np.ndarray) -> dict[int, np.ndarray]:
    """Group per-frame values by occlusion bin (from occlusion.occlusion_bins).
    Evaluate every metric per bin — the paper's key table is metrics x bins."""
    return {int(b): values[bins == b] for b in np.unique(bins) if b >= 0}