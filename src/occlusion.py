"""Occlusion severity measurement for hand-object interaction frames.

Occlusion is the central stratification variable of the project: we bin every
frame by how much the hand occludes the object (and vice versa), then ask
whether pose error and predicted confidence track it.

Inputs are per-frame segmentation masks. For HO-3D/DexYCB these can be
rendered from the ground-truth meshes + poses; for in-the-wild frames use
SAM/hand-segmenters. All functions are numpy, mask convention: bool (H, W).
"""

from __future__ import annotations
import numpy as np

def occlusion_fraction(target_mask: np.ndarray, occluder_mask: np.ndarray) -> float:
    """Fraction of the target's *visible-if-alone* region covered by the occluder.

    occ = |target ∩ occluder| / |target|

    Note: target_mask should be the AMODAL mask (full projected silhouette of
    the object from its GT pose, ignoring the hand). If you only have modal
    (visible) masks, use `amodal_from_pose` in ho3d_data.py to render one.
    """
    target_area = target_mask.sum()
    if target_area == 0:
        return float("nan")  # target not in frame; caller should drop frame
    return float((target_mask & occluder_mask).sum() / target_area)

def occlusion_bins(
    fractions: np.ndarray,
    edges: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 1.0),
) -> np.ndarray:
    """Assign each frame an occlusion-severity bin index.

    Default bins: none/low [0,.1), medium [.1,.3), high [.3,.5), severe [.5,1].
    NaN fractions get bin -1 (dropped by evaluation).
    """
    fractions = np.asarray(fractions)
    bins = np.digitize(fractions, edges[1:-1])
    bins = np.where(np.isnan(fractions), -1, bins)
    return bins.astype(int)

def truncation_fraction(mask: np.ndarray, border: int = 2) -> float:
    """Fraction of the mask touching the image border (out-of-frame proxy).

    Egocentric video loses hands/objects off-frame constantly; truncation is a
    second nuisance variable worth reporting alongside occlusion.
    """
    if mask.sum() == 0:
        return float("nan")
    edge = np.zeros_like(mask)
    edge[:border, :] = edge[-border:, :] = True
    edge[:, :border] = edge[:, -border:] = True
    return float((mask & edge).sum()/mask.sum())