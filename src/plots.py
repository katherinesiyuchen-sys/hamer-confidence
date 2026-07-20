"""Figures for the report: every plot the analysis needs, saved as PNG+PDF.

All functions take arrays already computed by metrics.py — no recomputation,
no dataset access — so they run anywhere (including on cached results).
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics

ESTIMATOR_LABELS = {
    "tta_ensemble": "TTA ensemble",
    "temporal_jitter": "Temporal jitter",
    "reprojection": "Reprojection (MediaPipe)",
    "learned_head": "Learned head",
}
BIN_LABELS = ["none/low", "medium", "high", "severe"]


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def sparsification_plot(scores_by_est: dict[str, np.ndarray], errors: np.ndarray,
                        out_dir: Path) -> None:
    """All estimators' sparsification curves + the oracle on one figure."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    oracle_drawn = False
    for name, s in scores_by_est.items():
        keep = ~np.isnan(s)
        if keep.sum() < 100:
            continue
        fr, curve, oracle, ause = metrics.sparsification_curve(s[keep], errors[keep])
        ax.plot(fr, curve, label=f"{ESTIMATOR_LABELS.get(name, name)} (AUSE {ause:.3f})")
        if not oracle_drawn:
            ax.plot(fr, oracle, "k--", lw=1, label="oracle (true error)")
            oracle_drawn = True
    ax.set_xlabel("fraction of most-uncertain frames removed")
    ax.set_ylabel("mean MPJPE of remaining frames (mm)")
    ax.set_title("Sparsification: does the score rank bad frames first?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "sparsification")


def score_error_scatter(scores_by_est: dict[str, np.ndarray], errors: np.ndarray,
                        out_dir: Path, max_points: int = 3000) -> None:
    """One scatter panel per estimator, with Spearman rho in the title."""
    live = [(k, v) for k, v in scores_by_est.items() if (~np.isnan(v)).sum() >= 100]
    if not live:
        return
    fig, axes = plt.subplots(1, len(live), figsize=(4 * len(live), 3.6), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (name, s) in zip(axes[0], live):
        keep = np.flatnonzero(~np.isnan(s))
        if len(keep) > max_points:
            keep = rng.choice(keep, max_points, replace=False)
        corr = metrics.score_error_correlation(s[keep], errors[keep])
        ax.scatter(s[keep], errors[keep], s=3, alpha=0.25)
        ax.set_title(f"{ESTIMATOR_LABELS.get(name, name)}\n"
                     f"Spearman ρ = {corr['spearman_rho']:.2f}", fontsize=9)
        ax.set_xlabel("uncertainty score")
        ax.set_ylabel("MPJPE (mm)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "score_vs_error")


def occlusion_stratified_bars(scores_by_est: dict[str, np.ndarray],
                              errors: np.ndarray, bins: np.ndarray,
                              out_dir: Path) -> None:
    """Two panels: (a) error rises with occlusion — the problem exists;
    (b) per-bin Spearman rho per estimator — who survives occlusion?"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    uniq = [b for b in np.unique(bins) if b >= 0]
    err_by_bin = [errors[bins == b] for b in uniq]
    ax1.bar([BIN_LABELS[b] if b < len(BIN_LABELS) else str(b) for b in uniq],
            [e.mean() for e in err_by_bin],
            yerr=[e.std() / np.sqrt(len(e)) for e in err_by_bin],
            color="#8C1515", alpha=0.85)
    ax1.set_ylabel("mean MPJPE (mm)")
    ax1.set_xlabel("occlusion severity")
    ax1.set_title("(a) Pose error vs. occlusion")
    ax1.grid(alpha=0.3, axis="y")

    width = 0.8 / max(len(scores_by_est), 1)
    x = np.arange(len(uniq))
    for i, (name, s) in enumerate(scores_by_est.items()):
        rhos = []
        for b in uniq:
            m = (bins == b) & ~np.isnan(s)
            rhos.append(metrics.score_error_correlation(s[m], errors[m])["spearman_rho"]
                        if m.sum() > 50 else np.nan)
        ax2.bar(x + i * width, rhos, width, label=ESTIMATOR_LABELS.get(name, name))
    ax2.set_xticks(x + 0.4 - width / 2)
    ax2.set_xticklabels([BIN_LABELS[b] if b < len(BIN_LABELS) else str(b) for b in uniq])
    ax2.set_ylabel("Spearman ρ (score vs error)")
    ax2.set_xlabel("occlusion severity")
    ax2.set_title("(b) Confidence quality vs. occlusion")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out_dir, "occlusion_stratified")


def filtering_plot(scores_by_est: dict[str, np.ndarray], errors: np.ndarray,
                   out_dir: Path) -> None:
    """Demonstration-filtering: mean error of kept frames vs keep-fraction."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    fracs = np.linspace(0.1, 1.0, 19)
    for name, s in scores_by_est.items():
        keep_mask = ~np.isnan(s)
        if keep_mask.sum() < 100:
            continue
        rows = metrics.filtering_experiment(s[keep_mask], errors[keep_mask],
                                            keep_fractions=tuple(fracs))
        ax.plot([r["keep_fraction"] for r in rows],
                [r["mean_error_mm"] for r in rows],
                marker="o", ms=3, label=ESTIMATOR_LABELS.get(name, name))
    ax.axhline(errors.mean(), color="k", ls=":", lw=1, label="no filtering")
    ax.set_xlabel("fraction of frames kept (most confident first)")
    ax.set_ylabel("mean MPJPE of kept frames (mm)")
    ax.set_title("Demonstration filtering: cleaner data by trusting the score")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "filtering")


def make_all(scores_npz: str | Path, out_dir: str | Path) -> None:
    """Load a scores.npz produced by run_eval.py and emit every figure."""
    from . import occlusion

    out_dir = Path(out_dir)
    with np.load(scores_npz, allow_pickle=True) as z:
        data = dict(z)
    errors = data["error"]
    bins = occlusion.occlusion_bins(data["occ"])
    ests = {k: data[k] for k in ESTIMATOR_LABELS if k in data}

    sparsification_plot(ests, errors, out_dir)
    score_error_scatter(ests, errors, out_dir)
    occlusion_stratified_bars(ests, errors, bins, out_dir)
    filtering_plot(ests, errors, out_dir)
    print(f"figures written to {out_dir}/")