from collections import deque
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class TemporalFeatures:
    palm_velocity_trend: float
    palm_acceleration_trend: float
    palm_stability_trend: float
    finger_motion_magnitude: float
    finger_consistency: float
    is_moving: bool
    is_stable: bool
    is_transitioning: bool
    is_pinched: bool
    pinch_duration: float
    confidence: float


class TemporalAnalyzer:
    VELOCITY_THRESHOLD = config.VELOCITY_THRESHOLD
    STABILITY_THRESHOLD = config.STABILITY_THRESHOLD
    PINCH_THRESHOLD = config.PINCH_THRESHOLD
    FINGER_MOTION_THRESHOLD = config.FINGER_MOTION_THRESHOLD

    def __init__(self, window_size: int = config.TEMPORAL_WINDOW_SIZE):
        self.window = deque(maxlen=window_size)
        self._pinch_start_time = None
        self._pinch_active = False
        self._pinch_enter_threshold = self.PINCH_THRESHOLD
        self._pinch_exit_threshold = self.PINCH_THRESHOLD * 1.2

    def reset(self):
        self.window.clear()
        self._pinch_start_time = None
        self._pinch_active = False

    def update(self, features):
        self.window.append(features)

        recent = list(self.window)[-5:]
        velocities = np.array([f.palm_velocity for f in recent], dtype=np.float32)
        accelerations = np.array([f.palm_acceleration for f in recent], dtype=np.float32)
        stabilities = np.array([f.palm_stability for f in recent], dtype=np.float32)

        if len(recent) > 0:
            weights = np.arange(1, len(recent) + 1, dtype=np.float32)
            weights /= np.sum(weights)
        else:
            weights = np.array([], dtype=np.float32)

        velocity_trend = float(np.dot(velocities, weights)) if len(velocities) else 0.0
        acceleration_trend = float(np.dot(accelerations, weights)) if len(accelerations) else 0.0
        stability_trend = float(np.dot(stabilities, weights)) if len(stabilities) else 1.0

        finger_motion = self._compute_finger_motion()
        finger_consistency = float(1.0 / (1.0 + finger_motion))

        if self._pinch_active:
            is_pinched = features.thumb_index_distance < self._pinch_exit_threshold
        else:
            is_pinched = features.thumb_index_distance < self._pinch_enter_threshold

        self._pinch_active = is_pinched
        pinch_duration = 0.0
        if is_pinched:
            if self._pinch_start_time is None:
                self._pinch_start_time = features.timestamp
            pinch_duration = float(features.timestamp - self._pinch_start_time)
        else:
            self._pinch_start_time = None

        is_moving = velocity_trend > self.VELOCITY_THRESHOLD
        is_stable = stability_trend > self.STABILITY_THRESHOLD
        is_transitioning = finger_motion > self.FINGER_MOTION_THRESHOLD

        confidence = float(np.clip((stability_trend + finger_consistency) / 2.0, 0.0, 1.0))

        return TemporalFeatures(
            palm_velocity_trend=velocity_trend,
            palm_acceleration_trend=acceleration_trend,
            palm_stability_trend=stability_trend,
            finger_motion_magnitude=finger_motion,
            finger_consistency=finger_consistency,
            is_moving=is_moving,
            is_stable=is_stable,
            is_transitioning=is_transitioning,
            is_pinched=is_pinched,
            pinch_duration=pinch_duration,
            confidence=confidence,
        )

    def _compute_finger_motion(self) -> float:
        if len(self.window) < 2:
            return 0.0

        tips = [4, 8, 12, 16, 20]
        displacements = []
        items = list(self.window)
        for prev, curr in zip(items[:-1], items[1:]):
            prev_points = prev.landmarks[tips, :2]
            curr_points = curr.landmarks[tips, :2]
            d = np.linalg.norm(curr_points - prev_points, axis=1)
            displacements.append(np.mean(d))

        return float(np.mean(displacements)) if displacements else 0.0
