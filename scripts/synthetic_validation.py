"""End-to-end pipeline validation on synthetic data (no dataset, no GPU).

Simulates the full experiment with a known ground truth so every analysis
stage can be verified before spending GPU-hours on the real thing:

  * GT hand trajectories: smooth random motion of 21 joints
  * occlusion: smooth random walk in [0,1] per frame
  * "HaMeR" predictions: GT + noise whose scale GROWS with occlusion,
    plus occasional catastrophic failures (tracking losses)
  * every uncertainty estimator computed through the REAL functions in
    src/uncertainty.py (not resampled numbers)
  * report + figures through the REAL metrics.py / plots.py

Expected outcome if the pipeline is correct:
  - error rises monotonically across occlusion bins  (planted)
  - all estimators get positive Spearman rho; learned head near the top
  - filtering curves slope down; AUSE well below random

Usage:  python scripts/synthetic_validation.py --out results_synth/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics, occlusion, plots, uncertainty  # noqa: E402

K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])  # synthetic camera


def smooth_walk(T: int, dim: int, step: float, rng) -> np.ndarray:
    """Cumulative Gaussian steps, then light smoothing — plausible motion."""
    x = np.cumsum(rng.normal(0, step, (T, dim)), axis=0)
    kernel = np.ones(5) / 5
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, "same"), 0, x)


def make_sequence(T: int, rng) -> dict:
    """One synthetic video sequence with GT, occlusion, and noisy predictions."""
    # GT: hand ~40cm from camera, joints in a ~10cm blob, smooth motion
    center = np.array([0.0, 0.0, 0.4]) + smooth_walk(T, 3, 0.002, rng)
    fingers = rng.normal(0, 0.04, (21, 3))
    articulation = smooth_walk(T, 63, 0.0015, rng).reshape(T, 21, 3)
    gt = center[:, None, :] + fingers[None] + articulation          # (T,21,3)

    # occlusion in [0,1], smooth, varied per sequence
    occ = np.clip(0.5 + smooth_walk(T, 1, 0.03, rng)[:, 0], 0, 1)

    # prediction noise: base 4mm, +25mm at full occlusion; 2% tracking losses
    sigma = 0.004 + 0.025 * occ ** 1.5                              # (T,)
    pred = gt + rng.normal(0, 1, gt.shape) * sigma[:, None, None]
    lost = rng.random(T) < 0.02
    pred[lost] += rng.normal(0, 0.05, (lost.sum(), 21, 3))          # blowups

    # TTA ensemble: members scatter with the same per-frame sigma (a robust
    # frame is insensitive to crop jitter; a fragile one scatters)
    tta = pred[:, None] + rng.normal(0, 1, (T, 8, 21, 3)) * (0.7 * sigma)[:, None, None, None]

    # independent 2D detector: project GT + own 3px noise, worse w/ occlusion
    proj = gt @ K.T
    det2d = proj[..., :2] / proj[..., 2:3] + rng.normal(0, 1, (T, 21, 2)) * (3 + 8 * occ)[:, None, None]

    # backbone "features": encode true difficulty + nuisance dims
    feat = np.concatenate([
        occ[:, None] + rng.normal(0, 0.15, (T, 1)),
        sigma[:, None] * 100 + rng.normal(0, 0.3, (T, 1)),
        rng.normal(0, 1, (T, 30)),
    ], axis=1)

    return dict(gt=gt, pred=pred, occ=occ, tta=tta, det2d=det2d, feat=feat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results_synth"))
    ap.add_argument("--sequences", type=int, default=40)
    ap.add_argument("--frames", type=int, default=250)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    rows = {k: [] for k in ["error", "occ", "tta_ensemble",
                            "temporal_jitter", "reprojection", "learned_head"]}
    feats, errs = [], []

    for _ in range(args.sequences):
        seq = make_sequence(args.frames, rng)
        err = metrics.mpjpe(seq["pred"], seq["gt"]) * 1000            # mm
        tj = uncertainty.temporal_jitter(seq["pred"])                  # real fn
        for t in range(args.frames):
            rows["error"].append(err[t])
            rows["occ"].append(seq["occ"][t])
            rows["tta_ensemble"].append(
                uncertainty.ensemble_disagreement(seq["tta"][t]))      # real fn
            rows["temporal_jitter"].append(tj[t])
            rows["reprojection"].append(uncertainty.reprojection_residual(
                seq["pred"][t], K, seq["det2d"][t]))                   # real fn
            rows["learned_head"].append(np.nan)                        # below
        feats.append(seq["feat"])
        errs.append(err)

    # learned head: real training loop on cached "features"
    import torch
    F = torch.tensor(np.concatenate(feats), dtype=torch.float32)
    E = torch.tensor(np.concatenate(errs), dtype=torch.float32)
    n_tr = int(0.7 * len(F))
    head = uncertainty.LearnedConfidenceHead(feat_dim=F.shape[1], hidden=64)
    head = uncertainty.train_confidence_head(head, F[:n_tr], E[:n_tr],
                                             epochs=30, device="cpu")
    with torch.no_grad():
        pred_err = head(F).numpy()
    lh = np.full(len(F), np.nan)
    lh[n_tr:] = pred_err[n_tr:]                    # honest: held-out only
    rows["learned_head"] = list(lh)

    data = {k: np.array(v) for k, v in rows.items()}
    np.savez(args.out / "scores.npz", **data)

    # ---- report (same structure as run_eval.stage_report) ----
    errors, bins = data["error"], occlusion.occlusion_bins(data["occ"])
    report = {"n_frames": int(len(errors)),
              "mean_mpjpe_mm": float(errors.mean()),
              "mean_err_by_occlusion_bin": {
                  str(b): float(errors[bins == b].mean())
                  for b in np.unique(bins) if b >= 0}}
    for est in ["tta_ensemble", "temporal_jitter", "reprojection", "learned_head"]:
        s = data[est]
        keep = ~np.isnan(s)
        s_k, e_k, b_k = s[keep], errors[keep], bins[keep]
        _, _, _, ause = metrics.sparsification_curve(s_k, e_k)
        report[est] = {
            "correlation": metrics.score_error_correlation(s_k, e_k),
            "ause": ause,
            "auroc_20mm": metrics.failure_auroc(s_k, e_k, 20.0),
            "filtering_keep50": metrics.filtering_experiment(
                s_k, e_k, keep_fractions=(0.5,))[0],
            "rho_by_occlusion_bin": {
                str(b): metrics.score_error_correlation(
                    s_k[b_k == b], e_k[b_k == b])["spearman_rho"]
                for b in np.unique(b_k) if b >= 0 and (b_k == b).sum() > 50},
        }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))

    # ---- figures through the real plotting module ----
    plots.make_all(args.out / "scores.npz", args.out / "figures")

    # ---- verdict ----
    print("\n=== VALIDATION SUMMARY ===")
    print(f"frames: {report['n_frames']}, mean MPJPE {report['mean_mpjpe_mm']:.1f}mm")
    print("error by occlusion bin:", {k: round(v, 1) for k, v in
                                      report["mean_err_by_occlusion_bin"].items()})
    ok = True
    binvals = list(report["mean_err_by_occlusion_bin"].values())
    if not all(b2 > b1 for b1, b2 in zip(binvals, binvals[1:])):
        ok = False
        print("!! error does not rise with occlusion — check occlusion model")
    for est in ["tta_ensemble", "temporal_jitter", "reprojection", "learned_head"]:
        rho = report[est]["correlation"]["spearman_rho"]
        print(f"{est:18s} rho={rho:+.3f}  AUSE={report[est]['ause']:.3f}  "
              f"AUROC={report[est]['auroc_20mm']:.3f}  "
              f"filter@50%: {report[est]['filtering_keep50']['mean_error_mm']:.1f}mm")
        if rho < 0.2:
            ok = False
            print(f"!! {est} barely correlates — investigate")
    print("\nPIPELINE " + ("VALIDATED ✓" if ok else "HAS ISSUES ✗"))


if __name__ == "__main__":
    main()