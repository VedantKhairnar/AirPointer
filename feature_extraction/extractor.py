from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HandFeatures:
    palm_x: float
    palm_y: float
    palm_z: float
    palm_velocity: float
    palm_acceleration: float
    palm_stability: float
    index_extended: bool
    middle_extended: bool
    ring_extended: bool
    pinky_extended: bool
    thumb_extended: bool
    thumb_index_distance: float
    finger_spread: float
    timestamp: float
    landmarks: np.ndarray


class FeatureExtractor:
    def __init__(self):
        self._last_pos: Optional[np.ndarray] = None
        self._last_vel = 0.0
        self._last_time: Optional[float] = None

    def reset(self):
        self._last_pos = None
        self._last_vel = 0.0
        self._last_time = None

    def extract_features(self, landmarks: np.ndarray, timestamp: float) -> HandFeatures:
        # Blend wrist and middle-MCP for a more stable cursor anchor.
        palm = (0.4 * landmarks[0, :3]) + (0.6 * landmarks[9, :3])

        if self._last_pos is None or self._last_time is None:
            velocity = 0.0
            acceleration = 0.0
        else:
            dt = max(timestamp - self._last_time, 1e-6)
            displacement = np.linalg.norm(palm - self._last_pos)
            velocity = float(displacement / dt)
            acceleration = float((velocity - self._last_vel) / dt)

        stability = float(1.0 / (1.0 + abs(acceleration)))

        index_extended = self._is_finger_extended(landmarks, 8, 5)
        middle_extended = self._is_finger_extended(landmarks, 12, 9)
        ring_extended = self._is_finger_extended(landmarks, 16, 13)
        pinky_extended = self._is_finger_extended(landmarks, 20, 17)
        thumb_extended = self._is_thumb_extended(landmarks)

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_dist_raw = float(np.linalg.norm(thumb_tip[:3] - index_tip[:3]))
        thumb_index_distance = pinch_dist_raw / 0.3

        fingertip_indices = [8, 12, 16, 20]
        mcp_indices = [5, 9, 13, 17]
        fingertip_points = landmarks[fingertip_indices, :2]
        mcp_points = landmarks[mcp_indices, :2]
        finger_spread = float(np.mean(np.linalg.norm(fingertip_points - mcp_points, axis=1)))

        self._last_pos = palm.copy()
        self._last_vel = velocity
        self._last_time = timestamp

        return HandFeatures(
            palm_x=float(palm[0]),
            palm_y=float(palm[1]),
            palm_z=float(palm[2]),
            palm_velocity=velocity,
            palm_acceleration=acceleration,
            palm_stability=stability,
            index_extended=index_extended,
            middle_extended=middle_extended,
            ring_extended=ring_extended,
            pinky_extended=pinky_extended,
            thumb_extended=thumb_extended,
            thumb_index_distance=thumb_index_distance,
            finger_spread=finger_spread,
            timestamp=timestamp,
            landmarks=landmarks.copy(),
        )

    @staticmethod
    def _is_finger_extended(landmarks: np.ndarray, tip_idx: int, mcp_idx: int) -> bool:
        wrist = landmarks[0, :3]
        tip_dist = np.linalg.norm(landmarks[tip_idx, :3] - wrist)
        mcp_dist = np.linalg.norm(landmarks[mcp_idx, :3] - wrist)
        return bool(tip_dist > (mcp_dist * 1.12))

    @staticmethod
    def _is_thumb_extended(landmarks: np.ndarray) -> bool:
        wrist = landmarks[0, :3]
        thumb_tip = landmarks[4, :3]
        thumb_mcp = landmarks[2, :3]
        tip_dist = np.linalg.norm(thumb_tip - wrist)
        mcp_dist = np.linalg.norm(thumb_mcp - wrist)
        return bool(tip_dist > (mcp_dist * 1.18))
