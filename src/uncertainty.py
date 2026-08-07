"""Per-frame confidence estimators for hand pose reconstruction.

Each estimator maps a frame (or short window) to a scalar uncertainty score.
Higher score = model probably wrong. The project's question: which score best
predicts actual 3D pose error, especially under occlusion?

Estimators implemented:
  1. ensemble_disagreement: variance across an ensemble of predictors
  2. mc_dropout: variance across stochastic forward passes
  3. temporal_jitter: frame-to-frame pose inconsistency
  4. reprojection_residual: 2D keypoint disagreement with an independent detector
  5. LearnedConfidenceHead: small MLP trained to regress error from features

All pose arrays follow HaMeR/MANO conventions:
  joints3d: (21, 3) in meters, root-relative unless noted.
"""

from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:  # torch only needed for mc_dropout and the learned head
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:  # numpy-only estimators still importable (e.g. on laptop)
    _TORCH = False

    class _Stub:  # minimal placeholder so class definitions below parse
        Module = object
    nn = _Stub()  # type: ignore

# 1. Ensemble disagreement

def ensemble_disagreement(joint_preds: np.ndarray) -> np.ndarray:
    """Mean per-joint std across ensemble members.

    joint_preds: (M, 21, 3) M ensemble members' 3D joints for one frame.
    An "ensemble" can be M different checkpoints, or the same model run on
    M augmented crops (test-time augmentation)

    (T, M, 21, 3) -> (T,). Batched version of ensemble_disagreement.
    """
    std = preds.std(axis=1)
    return np.linalg.norm(std, axis=-1).mean(axis=-1)


# 2. MC-dropout

def mc_dropout(model, image, n_samples: int = 10) -> tuple[float, np.ndarray]:
    """Variance over stochastic forward passes with dropout kept active.

    Works on any model with dropout layers (HaMeR's ViT has them). We flip
    only Dropout modules to train mode so batchnorm stats stay frozen.
    Returns (score, mean_joints) so the mean prediction can be reused.
    """
    with torch.no_grad():
        return _mc_dropout_impl(model, image, n_samples)


def _mc_dropout_impl(model, image, n_samples):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()
    preds = []
    for _ in range(n_samples):
        out = model(image)      # adapt to HaMeR's output dict
        preds.append(out["pred_keypoints_3d"].squeeze(0).cpu().numpy())
    preds = np.stack(preds)     # (n, 21, 3)

    for m in model.modules():   # restore
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.eval()
    return ensemble_disagreement(preds), preds.mean(axis=0)


# 3. Temporal jitter

def temporal_jitter(joints_seq: np.ndarray, window: int = 5) -> np.ndarray:
    """Per-frame inconsistency vs. a local median trajectory.

    joints_seq: (T, 21, 3). Real hands move smoothly; a reconstruction that
    teleports between frames is a strong unreliability signal (and free —
    needs no extra model). Returns (T,) scores.

    Uses distance to the median-filtered trajectory rather than raw velocity
    so genuinely fast motion is not punished as much as inconsistency.
    """
    T = joints_seq.shape[0] 
    half = window // 2
    scores = np.empty(T)

    if T >= window:
        win = sliding_window_view(joints_seq, window, axis=0)    # (T-w+1, 21, 3, w)
        med = np.median(win, axis=-1)
        scores[half:T-half] = np.linalg.norm(
            joints_seq[half:T-half] - med, axis=-1).mean(axis=-1)

    for t in list(range(min(half, T))) + list(range(max(T-half, 0), T)):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        med = np.median(joints_seq[lo:hi], axis=0)
        scores[t] = np.linalg.norm(joints_seq[t] - med, axis=-1).mean()

    return scores


# 4. Reprojection residual against an independent 2D detector

def reprojection_residual(joints3d_cam: np.ndarray, K: np.ndarray, 
                          joints2d_detector: np.ndarray, 
                          detector_conf: np.ndarray | None = None,) -> np.ndarray:
    """Project the 3D prediction and compare with an independent 2D detector.

    joints3d_cam: (21, 3) in CAMERA frame (meters).
    K: (3, 3) camera intrinsics.
    joints2d_detector: (21, 2) pixels from e.g. MediaPipe a different model
        family, so its errors are decorrelated from HaMeR's.
    detector_conf: (21,) optional per-joint weights.

    Two models agreeing is evidence both are right; disagreement flags trouble.

    (T,21,3),(3,3),(T,21,2)[,(T,21)] -> (T,). Batched reprojection residual.
    """
    proj = joints3d_cam @ K.T       # (21, 3)
    proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)
    err = np.linalg.norm(proj - joints2d_detector, axis=-1)   # (21,) px
    if detector_conf is not None:
        w = detector_conf / max(detector_conf.sum(), 1e-6)
        return (err * w).sum()
    return err.mean()


# 5. Learned confidence head

class LearnedConfidenceHead(nn.Module):
    """MLP: frozen backbone features -> predicted pose error (mm).

    Train with an L1 loss against the *actual* MPJPE of the base model on
    frames with ground truth. At test time its output IS the confidence score.
    This is the linear-probe idea from self-supervised learning: the backbone
    stays frozen; only this head trains. Keep it small to avoid overfitting.
    """

    def __init__(self, feat_dim: int = 1024, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),        # error is nonnegative
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats).squeeze(-1)


def train_confidence_head(
    head: LearnedConfidenceHead,
    feats: torch.Tensor,      # (N, feat_dim) cached backbone features
    errors: torch.Tensor,     # (N,) actual per-frame MPJPE in mm
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 256,
    device: str = "cuda",
) -> LearnedConfidenceHead:
    """Standard supervised regression loop. Features are precomputed/cached
    (one backbone pass over the dataset) so this trains in minutes on 1 GPU."""
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(feats, errors)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    head.train()
    for epoch in range(epochs):
        total = 0.0
        for f, e in loader:
            f, e = f.to(device), e.to(device)
            pred = head(f)
            loss = nn.functional.l1_loss(pred, e)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(f)
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1:3d}  L1 {total/len(ds):.2f} mm")
    head.eval()
    return head