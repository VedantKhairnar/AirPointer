"""
Perception module for AirPointer
- Handles camera input
- Runs MediaPipe Hands
- Outputs 21 hand landmarks per frame
"""

from .camera_handler import CameraHandler
from .hand_detector import HandDetector

__all__ = ['CameraHandler', 'HandDetector']
