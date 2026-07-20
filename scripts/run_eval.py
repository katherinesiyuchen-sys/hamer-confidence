"""End-to-end evaluation: every uncertainty estimator vs. actual error,
overall and stratified by occlusion severity.

Stages (each cached, so you can rerun any stage alone):
  1. inference  — HaMeR on every HO-3D frame -> cached joints/features
  2. scores     — compute all uncertainty scores from the cache
  3. report     — correlations, sparsification/AUSE, AUROC, filtering table,
                  all overall AND per occlusion bin

Usage:
  python scripts/run_eval.py --ho3d /path/to/HO3D_v3 --out results/ \
      --stage all --max-frames 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics, occlusion, uncertainty  # noqa: E402
from src.ho3d_data import HO3D  # noqa: E402

ESTIMATORS = ["tta_ensemble", "temporal_jitter", "reprojection", "learned_head"]

def stage_inference(args) -> None:
    """HaMeR over the dataset; cache joints3d + backbone features per frame."""
    import cv2
    from src.hamer_wrapper import HamerPredictor, cache_predictions, load_cached

    ds = HO3D(args.ho3d, split="train")
    predictor = HamerPredictor()
    n = min(len(ds), args.max_frames)
    print(f"running HaMeR on {n} frames")
    for i in range(n):
        fr = ds[i]
        key = f"{fr.seq}_{fr.idx:05d}"
        if load_cached(args.out / "preds", key) is not None:
            continue
        img = cv2.imread(str(fr.image_path))
        # bbox from GT joint projection (generous margin); swap in a hand
        # detector for a fully GT-free pipeline later
        pts = fr.joints3d_cam @ fr.K.T
        pts = pts[:, :2] / pts[:, 2:3]
        x1, y1 = pts.min(0) - 60
        x2, y2 = pts.max(0) + 60
        res = predictor.predict(img, np.array([x1, y1, x2, y2]))
        res["tta"] = predictor.predict_tta(img, np.array([x1, y1, x2, y2]), n=args.tta)
        cache_predictions(args.out / "preds", key, res)
        if i % 500 == 0:
            print(f"  {i}/{n}")


def stage_scores(args) -> None:
    """Assemble per-frame arrays: error, occlusion bin, every score."""
    from src.hamer_wrapper import load_cached

    ds = HO3D(args.ho3d, split="train")
    seqs = ds.sequences()

    rows: dict[str, list] = {k: [] for k in
                             ["key", "error", "occ", *ESTIMATORS]}
    feats, errs_for_head = [], []

    for seq, idxs in seqs.items():
        joints_seq, cached_seq, frames = [], [], []
        for i in idxs[: args.max_frames]:
            fr = ds[i]
            c = load_cached(args.out / "preds", f"{fr.seq}_{fr.idx:05d}")
            if c is None or np.isnan(fr.joints3d_cam).any():
                continue
            joints_seq.append(c["joints3d"])
            cached_seq.append(c)
            frames.append(fr)
        if not frames:
            continue

        joints_seq = np.stack(joints_seq)
        tj = uncertainty.temporal_jitter(joints_seq)

        for t, (fr, c) in enumerate(zip(frames, cached_seq)):
            err = metrics.mpjpe(c["joints3d"][None], fr.joints3d_cam[None])[0] * 1000
            rows["key"].append(f"{fr.seq}_{fr.idx:05d}")
            rows["error"].append(err)
            # occlusion fraction: precomputed masks expected at out/masks/
            occ_p = args.out / "masks" / f"{fr.seq}_{fr.idx:05d}.npz"
            if occ_p.exists():
                with np.load(occ_p) as z:
                    rows["occ"].append(occlusion.occlusion_fraction(
                        z["obj_amodal"], z["hand"]))
            else:
                rows["occ"].append(np.nan)
            rows["tta_ensemble"].append(
                uncertainty.ensemble_disagreement(c["tta"]) if "tta" in c else np.nan)
            rows["temporal_jitter"].append(tj[t])
            rows["reprojection"].append(np.nan)   # fill in after wiring MediaPipe
            rows["learned_head"].append(np.nan)   # filled after head training below
            if c.get("feature") is not None:
                feats.append(c["feature"])
                errs_for_head.append(err)

    # train the learned head on a split of the cached features
    if feats:
        import torch
        F = torch.tensor(np.stack(feats), dtype=torch.float32)
        E = torch.tensor(np.array(errs_for_head), dtype=torch.float32)
        n_tr = int(0.7 * len(F))
        head = uncertainty.LearnedConfidenceHead(feat_dim=F.shape[1])
        head = uncertainty.train_confidence_head(
            head, F[:n_tr], E[:n_tr],
            device="cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            preds = head(F.to(next(head.parameters()).device)).cpu().numpy()
        # note: only the held-out 30% is a fair eval for the learned head —
        # stage_report reads head_split to respect that
        rows["learned_head"] = list(preds)
        np.save(args.out / "head_split.npy", np.array([n_tr]))

    np.savez(args.out / "scores.npz",
             **{k: np.array(v) for k, v in rows.items() if k != "key"},
             key=np.array(rows["key"]))
    print(f"wrote {args.out/'scores.npz'} with {len(rows['error'])} frames")


def stage_report(args) -> None:
    with np.load(args.out / "scores.npz", allow_pickle=True) as z:
        data = dict(z)
    errors = data["error"]
    bins = occlusion.occlusion_bins(data["occ"])
    report: dict = {"n_frames": int(len(errors)),
                    "mean_mpjpe_mm": float(np.nanmean(errors))}

    for est in ESTIMATORS:
        s = data[est]
        keep = ~np.isnan(s)
        if est == "learned_head" and (args.out / "head_split.npy").exists():
            n_tr = int(np.load(args.out / "head_split.npy")[0])
            keep[:n_tr] = False              # eval on held-out only
        if keep.sum() < 100:
            continue
        s_k, e_k, b_k = s[keep], errors[keep], bins[keep]
        entry = {
            "correlation": metrics.score_error_correlation(s_k, e_k),
            "auroc_20mm": metrics.failure_auroc(s_k, e_k, 20.0),
            "filtering": metrics.filtering_experiment(s_k, e_k),
        }
        _, _, _, ause = metrics.sparsification_curve(s_k, e_k)
        entry["ause"] = ause
        entry["per_occlusion_bin"] = {
            str(b): metrics.score_error_correlation(s_k[b_k == b], e_k[b_k == b])
            for b in np.unique(b_k) if b >= 0 and (b_k == b).sum() > 50
        }
        report[est] = entry

    out_path = args.out / "report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ho3d", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--stage", choices=["inference", "scores", "report", "all"],
                   default="all")
    p.add_argument("--max-frames", type=int, default=20000)
    p.add_argument("--tta", type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage in ("inference", "all"):
        stage_inference(args)
    if args.stage in ("scores", "all"):
        stage_scores(args)
    if args.stage in ("report", "all"):
        stage_report(args)


if __name__ == "__main__":
    main()