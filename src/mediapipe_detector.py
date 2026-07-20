"""Independent 2D hand keypoint detector (MediaPipe Hands).

Used by uncertainty.reprojection_residual: MediaPipe is a different model
family from HaMeR (lightweight CNN vs ViT-H, different training data), so
their errors are largely decorrelated — agreement between the two is evidence
of a reliable frame, disagreement flags trouble.

MediaPipe returns 21 landmarks in the SAME ordering as MANO/HaMeR
(wrist, thumb 1-4, index 1-4, middle 1-4, ring 1-4, pinky 1-4),
so no joint remapping is needed.
"""

from __future__ import annotations
import numpy as np

class MediaPipeHand2D:
    def __init__(self, min_confidence: float = 0.3):
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,    # per-frame, no temporal tracking:
            max_num_hands=2,    # keeps it an INDEPENDENT signal
            min_detection_confidence=min_confidence,
        )

    def detect(self, image_rgb: np.ndarray, want_right: bool = True):
        """Detect 2D hand keypoints in a full RGB frame.

        Returns (joints2d (21,2) pixels, per_joint_conf (21,)) or (None, None)
        if no matching hand found. MediaPipe gives handedness per detection;
        note its 'Right' label assumes an unmirrored image.
        """
        H, W = image_rgb.shape[:2]
        res = self._hands.process(image_rgb)
        if not res.multi_hand_landmarks:
            return None, None

        best = None
        for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            label_right = handed.classification[0].label == "Right"
            score = handed.classification[0].score
            if label_right == want_right:
                if best is None or score > best[1]:
                    best = (lm, score)
        if best is None:  # fall back to highest-score detection of any hand
            pairs = zip(res.multi_hand_landmarks, res.multi_handedness)
            best = max(((lm, h.classification[0].score) for lm, h in pairs),
                       key=lambda t: t[1])

        lm = best[0]
        pts = np.array([[p.x * W, p.y * H] for p in lm.landmark], dtype=np.float32)
        # MediaPipe Hands has no per-joint confidence; use detection score
        # replicated per joint (visibility field is unused by this solution).
        conf = np.full(21, best[1], dtype=np.float32)
        return pts, conf

    def close(self):
        self._hands.close()