from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import config

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None
    python = None
    vision = None


class HandDetector:
    def __init__(
        self,
        model_path: str = config.MEDIAPIPE_MODEL_PATH,
        max_hands: int = config.MEDIAPIPE_MAX_HANDS,
    ):
        self.model_path = model_path
        self.max_hands = max_hands
        self.landmarker = None
        self._connectors = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]
        self._initialize()

    def _initialize(self):
        if mp is None or python is None or vision is None:
            raise ImportError("mediapipe with tasks API is required")

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {model_file}")

        base_options = python.BaseOptions(model_asset_path=str(model_file))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.max_hands,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        if self.landmarker is None:
            return None, None

        frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect(mp_image)

        if not results.hand_landmarks:
            return None, None

        landmarks = results.hand_landmarks[0]
        data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        return data, None

    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        if landmarks is None:
            return frame

        height, width = frame.shape[:2]
        points = []
        for lm in landmarks:
            x = int(np.clip(lm[0], 0.0, 1.0) * width)
            y = int(np.clip(lm[1], 0.0, 1.0) * height)
            points.append((x, y))
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        for start, end in self._connectors:
            cv2.line(frame, points[start], points[end], (255, 200, 0), 2)
        return frame
