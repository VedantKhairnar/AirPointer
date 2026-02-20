"""
Temporal Analysis for AirPointer
Maintains sliding window of features, computes deltas and stability
"""

import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class TemporalFeatures:
    """Container for temporal features"""
    # Current frame features
    current_features: object  # HandFeatures
    
    # Motion history (10-20 frame window)
    palm_velocity_trend: float  # Average velocity over window
    palm_acceleration_trend: float  # Average acceleration
    palm_stability_trend: float  # Average stability
    
    # Finger motion history
    finger_motion_magnitude: float  # How much fingers moved
    finger_consistency: float  # How consistent finger movement
    
    # Motion type inference
    is_moving: bool  # Palm moving significantly
    is_stable: bool  # Palm relatively stable
    is_transitioning: bool  # Between states
    
    # Pinch tracking
    is_pinched: bool  # Thumb-index distance < threshold
    pinch_duration: float  # How long pinch has been held
    
    # Uncertainty
    confidence: float  # How confident are we in current inference


class TemporalAnalyzer:
    def __init__(self, window_size=15, fps=30):
        """
        Initialize temporal analyzer.
        
        Args:
            window_size: Number of frames to keep in sliding window
            fps: Frames per second (for duration calculations)
        """
        self.window_size = window_size
        self.fps = fps
        self.feature_history = deque(maxlen=window_size)
        
        # Thresholds
        self.VELOCITY_THRESHOLD = 0.01  # Movement threshold
        self.STABILITY_THRESHOLD = 0.7  # Stability score threshold
        self.PINCH_THRESHOLD = 0.3  # Thumb-index distance threshold
        self.FINGER_MOTION_THRESHOLD = 0.02
    
    def update(self, features):
        """
        Update temporal analyzer with new frame features.
        
        Args:
            features: HandFeatures object
        
        Returns:
            TemporalFeatures object
        """
        self.feature_history.append(features)
        
        # Compute temporal statistics
        if len(self.feature_history) < 2:
            # Not enough history, return minimal temporal features
            return TemporalFeatures(
                current_features=features,
                palm_velocity_trend=features.palm_velocity,
                palm_acceleration_trend=features.palm_acceleration,
                palm_stability_trend=features.palm_stability,
                finger_motion_magnitude=0,
                finger_consistency=0,
                is_moving=features.palm_velocity > self.VELOCITY_THRESHOLD,
                is_stable=features.palm_stability > self.STABILITY_THRESHOLD,
                is_transitioning=False,
                is_pinched=features.thumb_index_distance < self.PINCH_THRESHOLD,
                pinch_duration=0,
                confidence=0.5
            )
        
        # Convert history to array for analysis
        history = list(self.feature_history)
        
        # Palm motion trends
        velocities = np.array([f.palm_velocity for f in history])
        accelerations = np.array([f.palm_acceleration for f in history])
        stabilities = np.array([f.palm_stability for f in history])
        
        palm_velocity_trend = float(np.mean(velocities))
        palm_acceleration_trend = float(np.mean(accelerations))
        palm_stability_trend = float(np.mean(stabilities))
        
        # Motion detection
        is_moving = palm_velocity_trend > self.VELOCITY_THRESHOLD
        is_stable = palm_stability_trend > self.STABILITY_THRESHOLD
        is_transitioning = not is_moving and not is_stable
        
        # Finger motion consistency
        finger_motions = self._compute_finger_motion(history)
        finger_motion_magnitude = float(np.mean(finger_motions))
        finger_consistency = float(1.0 - np.std(finger_motions) / (np.mean(finger_motions) + 1e-6))
        
        # Pinch tracking
        pinch_distances = np.array([f.thumb_index_distance for f in history])
        is_pinched = float(np.mean(pinch_distances)) < self.PINCH_THRESHOLD
        
        # Pinch duration (how many consecutive frames pinched)
        pinch_duration = 0.0
        if is_pinched:
            for f in reversed(history):
                if f.thumb_index_distance < self.PINCH_THRESHOLD:
                    pinch_duration += 1.0 / self.fps
                else:
                    break
        
        # Overall confidence
        confidence = min(1.0, len(self.feature_history) / self.window_size)
        
        return TemporalFeatures(
            current_features=features,
            palm_velocity_trend=palm_velocity_trend,
            palm_acceleration_trend=palm_acceleration_trend,
            palm_stability_trend=palm_stability_trend,
            finger_motion_magnitude=finger_motion_magnitude,
            finger_consistency=finger_consistency,
            is_moving=is_moving,
            is_stable=is_stable,
            is_transitioning=is_transitioning,
            is_pinched=is_pinched,
            pinch_duration=pinch_duration,
            confidence=confidence
        )
    
    def _compute_finger_motion(self, history):
        """Compute finger motion magnitude for each frame"""
        finger_motions = []
        
        for i in range(len(history)):
            if i == 0:
                finger_motions.append(0)
            else:
                prev = history[i - 1]
                curr = history[i]
                
                # Motion of key finger tips
                motion = 0
                finger_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
                
                for idx in finger_indices:
                    prev_lm = prev.landmarks[idx]
                    curr_lm = curr.landmarks[idx]
                    motion += np.linalg.norm(curr_lm - prev_lm)
                
                finger_motions.append(motion / len(finger_indices))
        
        return finger_motions
