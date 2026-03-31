"""
MediaPipe Hand Detector for AirPointer
Extracts 21 hand landmarks from frames using MediaPipe Tasks API

This implementation uses the MediaPipe Tasks API (mediapipe>=0.10.0) which requires
the hand_landmarker.task model file. The model provides real-time hand landmark
detection with 21 keypoints per hand.

Model download:
  curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o hand_landmarker.task
"""

import mediapipe as mp
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:
    def __init__(self, max_num_hands=1, model_path="models/hand_landmarker.task"):
        """
        Initialize MediaPipe Tasks HandLandmarker.
        Args:
            max_num_hands: Maximum number of hands to detect
            model_path: Path to hand landmark model (optional, uses default if None)
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                              num_hands=max_num_hands)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_width = None
        self.frame_height = None
    
    def detect(self, frame):
        """
        Detect hand landmarks in frame using mediapipe.tasks.
        Args:
            frame: Input frame (numpy array)
        Returns:
            landmarks: List of 21 landmarks (each is [x, y, z]) or None if no hand detected
            confidence: Detection confidence or None
        """
        if self.frame_width is None:
            self.frame_height, self.frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand = results.hand_landmarks[0]
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand]
            # tasks API does not provide handedness/confidence directly
            confidence = None
            return np.array(landmarks), confidence
        return None, None
    
    def draw_landmarks(self, frame, landmarks):
        """
        Draw hand landmarks on frame (simple circles).
        Args:
            frame: Input frame
            landmarks: 21 landmarks
        Returns:
            Annotated frame
        """
        if landmarks is None:
            return frame
        h, w = frame.shape[:2]
        for i in range(21):
            x = int(landmarks[i][0] * w)
            y = int(landmarks[i][1] * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        return frame
