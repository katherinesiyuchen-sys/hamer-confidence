"""Thin wrapper around HaMeR (https://github.com/geopavlakos/hamer).

Install per their README (conda env + pretrained checkpoint download), then
this module gives the two calls the rest of the project needs:

  predict(image_bgr, bbox)   -> joints3d (21,3), backbone feature, mano params
  predict_tta(image_bgr, bbox, n) -> ensemble of predictions via test-time aug

We cache predictions + features to disk so the (slow) ViT-H backbone runs
once per frame; every uncertainty estimator afterwards reads the cache.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

class HamerPredictor:
    def __init__(self, device: str = "cuda"):
        # deferred import: hamer must be installed & checkpoints downloaded
        from hamer.models import load_hamer, DEFAULT_CHECKPOINT

        self.model, self.model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray, bbox_xyxy: np.ndarray, right: bool = True) -> dict:
        """Run HaMeR on one hand crop.

        Returns dict with:
          joints3d   (21,3) camera-frame joints (m)
          feature    (D,)   pooled ViT backbone feature (for the learned head)
          mano_pose  (48,)  axis-angle MANO pose
        """
        from hamer.datasets.vitdet_dataset import ViTDetDataset

        ds = ViTDetDataset(self.model_cfg, image_bgr, np.asarray([bbox_xyxy]), np.asarray([int(right)]),)
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=1)))
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = self.model(batch)

        return {
            "joints3d": out["pred_keypoints_3d"][0].cpu().numpy(),
            "feature": out["conditioning_feats"][0].float().cpu().numpy()
            if "conditioning_feats" in out else None,
            "mano_pose": out["pred_mano_params"]["hand_pose"][0].cpu().numpy().reshape(-1),
        }

    @torch.no_grad()
    def predict_tta(self, image_bgr: np.ndarray, bbox_xyxy: np.ndarray,
                    n: int = 8, jitter: float = 0.05, right: bool = True) -> np.ndarray:
        """Test-time-augmentation ensemble: rerun with jittered bboxes.

        Cheap stand-in for a checkpoint ensemble. Returns (n, 21, 3).
        Scale/shift jitter changes the crop the ViT sees; a robust prediction
        is invariant to it, a fragile one scatters.
        """
        x1, y1, x2, y2 = bbox_xyxy
        w, h = x2-x1, y2-y1
        rng = np.random.default_rng(0)
        preds = []
        for i in range(n):
            if i == 0:
                bb = bbox_xyxy
            else:
                dx, dy = rng.normal(0, jitter, 2) * (w, h)
                ds_ = 1 + rng.normal(0, jitter)
                cx, cy = (x1 + x2)/2 + dx, (y1 + y2)/2 + dy
                bb = np.array([cx - w*ds_/2, cy - h*ds_/2, cx + w*ds_/2, cy + h*ds_/2])
            preds.append(self.predict(image_bgr, bb, right)["joints3d"])
        return np.stack(preds)


def cache_predictions(pred_dir: str | Path, frame_key: str, result: dict) -> None:
    """np.savez one frame's outputs; key = f'{seq}_{idx:05d}'."""
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_dir/f"{frame_key}.npz", **{k: v for k, v in result.items() if v is not None})


def load_cached(pred_dir: str | Path, frame_key: str) -> dict | None:
    p = Path(pred_dir) / f"{frame_key}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return dict(z)