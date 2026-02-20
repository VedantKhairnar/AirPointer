"""
Feature Extraction for AirPointer
Converts 21 hand landmarks to interpretable physical features
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class HandFeatures:
    """Container for extracted hand features"""
    # Palm
    palm_x: float
    palm_y: float
    palm_z: float
    
    # Motion
    palm_velocity: float
    palm_acceleration: float
    palm_stability: float
    
    # Finger states (0=bent, 1=extended)
    index_extended: bool
    middle_extended: bool
    ring_extended: bool
    pinky_extended: bool
    thumb_extended: bool
    
    # Pinch distance (normalized 0-1)
    thumb_index_distance: float
    
    # Relative finger motion
    finger_spread: float  # Overall finger spread
    
    # Timestamp
    timestamp: float
    
    # Raw landmarks
    landmarks: np.ndarray


class FeatureExtractor:
    def __init__(self):
        """Initialize feature extractor"""
        # Landmark indices from MediaPipe Hands
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_FINGER_TIP = 8
        self.MIDDLE_FINGER_TIP = 12
        self.RING_FINGER_TIP = 16
        self.PINKY_TIP = 20
        
        # Mid-knuckles for finger extension detection
        self.THUMB_MCP = 2
        self.INDEX_MCP = 5
        self.MIDDLE_MCP = 9
        self.RING_MCP = 13
        self.PINKY_MCP = 17
        
        # PIP joints for finger extension
        self.INDEX_PIP = 6
        self.MIDDLE_PIP = 10
        self.RING_PIP = 14
        self.PINKY_PIP = 18
        
        self.prev_palm_pos = None
        self.prev_velocity = 0
    
    def extract_features(self, landmarks, timestamp):
        """
        Extract physical features from landmarks.
        
        Args:
            landmarks: (21, 3) array of hand landmarks [x, y, z]
            timestamp: Current timestamp (seconds)
        
        Returns:
            HandFeatures object
        """
        # Palm center (average of wrist and base of fingers)
        palm_x = landmarks[self.WRIST, 0]
        palm_y = landmarks[self.WRIST, 1]
        palm_z = landmarks[self.WRIST, 2]
        
        # Compute palm motion
        current_pos = np.array([palm_x, palm_y, palm_z])
        
        if self.prev_palm_pos is not None:
            # Velocity (Euclidean distance)
            palm_velocity = np.linalg.norm(current_pos - self.prev_palm_pos)
            
            # Acceleration (change in velocity)
            palm_acceleration = abs(palm_velocity - self.prev_velocity)
            
            # Stability (low acceleration = stable)
            palm_stability = 1.0 / (1.0 + palm_acceleration * 10)
        else:
            palm_velocity = 0
            palm_acceleration = 0
            palm_stability = 1.0
        
        self.prev_palm_pos = current_pos
        self.prev_velocity = palm_velocity
        
        # Finger extension detection (compare tip to MCP joint)
        def is_extended(tip_idx, mcp_idx, pip_idx):
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            pip = landmarks[pip_idx]
            # Extended if tip is far from MCP (distance metric)
            dist_to_mcp = np.linalg.norm(tip - mcp)
            dist_pip_to_mcp = np.linalg.norm(pip - mcp)
            return dist_to_mcp > dist_pip_to_mcp * 0.8
        
        index_extended = is_extended(self.INDEX_FINGER_TIP, self.INDEX_MCP, self.INDEX_PIP)
        middle_extended = is_extended(self.MIDDLE_FINGER_TIP, self.MIDDLE_MCP, self.MIDDLE_PIP)
        ring_extended = is_extended(self.RING_FINGER_TIP, self.RING_MCP, self.RING_PIP)
        pinky_extended = is_extended(self.PINKY_TIP, self.PINKY_MCP, self.PINKY_PIP)
        
        # Thumb extension (different heuristic)
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_mcp = landmarks[self.THUMB_MCP]
        wrist = landmarks[self.WRIST]
        thumb_extended = np.linalg.norm(thumb_tip - wrist) > np.linalg.norm(thumb_mcp - wrist) * 0.6
        
        # Thumb-Index pinch distance
        thumb_idx_dist = np.linalg.norm(
            landmarks[self.THUMB_TIP] - landmarks[self.INDEX_FINGER_TIP]
        )
        # Normalize: 0 = pinched, 1 = far apart (assuming max ~0.3)
        thumb_index_distance = min(1.0, thumb_idx_dist / 0.3)
        
        # Finger spread (std dev of finger positions)
        finger_tips = np.array([
            landmarks[self.THUMB_TIP],
            landmarks[self.INDEX_FINGER_TIP],
            landmarks[self.MIDDLE_FINGER_TIP],
            landmarks[self.RING_FINGER_TIP],
            landmarks[self.PINKY_TIP]
        ])
        finger_spread = float(np.std(finger_tips, axis=0).mean())
        
        return HandFeatures(
            palm_x=float(palm_x),
            palm_y=float(palm_y),
            palm_z=float(palm_z),
            palm_velocity=float(palm_velocity),
            palm_acceleration=float(palm_acceleration),
            palm_stability=float(palm_stability),
            index_extended=index_extended,
            middle_extended=middle_extended,
            ring_extended=ring_extended,
            pinky_extended=pinky_extended,
            thumb_extended=thumb_extended,
            thumb_index_distance=float(thumb_index_distance),
            finger_spread=float(finger_spread),
            timestamp=timestamp,
            landmarks=landmarks
        )
