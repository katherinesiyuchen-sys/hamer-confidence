"""HO-3D v3 dataset access: frames, ground truth, and occlusion masks.

Download (registration required):
  https://github.com/shreyashampali/ho3d  ->  OneDrive links in the README.
Layout expected below (HO3D_v3 root):

  HO3D_v3/
    train/<seq>/rgb/*.jpg
    train/<seq>/meta/*.pkl  
    evaluation/<seq>/...

Object CAD models come from the YCB models release (same README).
GT hand joints use MANO ordering (21 joints, wrist = 0).
"""

from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# OpenGL->OpenCV convention flip used by HO-3D annotations
COORD_FLIP = np.array([1.0, -1.0, -1.0], dtype=np.float32)


@dataclass
class Frame:
    image_path: Path
    K: np.ndarray    # (3,3) intrinsics
    joints3d_cam: np.ndarray    # (21,3) GT hand joints, camera frame, meters
    obj_rot: np.ndarray    # (3,) axis-angle, object->camera
    obj_trans: np.ndarray    # (3,) meters
    obj_name: str
    seq: str
    idx: int


class HO3D:
    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root) / split
        self.samples: list[tuple[str, str]] = []     # (seq, frame_id)
        for seq_dir in sorted(self.root.iterdir()):
            rgb = seq_dir / "rgb"
            if not rgb.is_dir():
                continue
            for img in sorted(rgb.glob("*.jpg")):
                self.samples.append((seq_dir.name, img.stem))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Frame:
        seq, fid = self.samples[i]
        meta_path = self.root / seq / "meta" / f"{fid}.pkl"
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        joints = meta["handJoints3D"]
        # evaluation split hides full hand GT (only wrist); guard for that
        if joints is None or joints.ndim == 1:
            joints3d = np.full((21, 3), np.nan, dtype=np.float32)
        else:
            joints3d = joints.astype(np.float32) * COORD_FLIP

        return Frame(
            image_path=self.root / seq / "rgb" / f"{fid}.jpg",
            K=meta["camMat"].astype(np.float32),
            joints3d_cam=joints3d,
            obj_rot=meta["objRot"].reshape(3).astype(np.float32),
            obj_trans=(meta["objTrans"].astype(np.float32) * COORD_FLIP),
            obj_name=meta["objName"],
            seq=seq,
            idx=int(fid),
        )

    def sequences(self) -> dict[str, list[int]]:
        """Map sequence name -> sorted list of sample indices (for temporal
        estimators, which need contiguous frames of one sequence)."""
        out: dict[str, list[int]] = {}
        for i, (seq, _) in enumerate(self.samples):
            out.setdefault(seq, []).append(i)
        return out


def amodal_object_mask(frame: Frame, obj_mesh_verts: np.ndarray,
                       obj_mesh_faces: np.ndarray, img_hw: tuple[int, int]) -> np.ndarray:
    """Render the object's full silhouette from GT pose (amodal mask).

    Uses trimesh/pyrender offscreen. The AMODAL mask (object as if the hand
    were invisible) is what occlusion_fraction() needs as target_mask; the
    hand mask (render MANO the same way, or use a segmenter) is the occluder.
    """
    import cv2
    import trimesh
    import pyrender

    R, _ = cv2.Rodrigues(frame.obj_rot)
    verts = obj_mesh_verts @ R.T + frame.obj_trans
    mesh = trimesh.Trimesh(vertices=verts, faces=obj_mesh_faces, process=False)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    cam = pyrender.IntrinsicsCamera(
        fx=frame.K[0, 0], fy=frame.K[1, 1], cx=frame.K[0, 2], cy=frame.K[1, 2])
    # OpenCV cam looks down +z; pyrender down -z — rotate 180deg about x
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    scene.add(cam, pose=pose)

    r = pyrender.OffscreenRenderer(img_hw[1], img_hw[0])
    _, depth = r.render(scene)
    r.delete()
    return depth > 0