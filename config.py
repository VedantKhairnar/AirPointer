"""
Configuration file for AirPointer
Centralized settings for easy tuning without modifying code
"""

# Camera settings
CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# MediaPipe Tasks API settings
# Note: MediaPipe now uses the Tasks API with hand_landmarker.task model
MEDIAPIPE_MAX_HANDS = 1  # Maximum number of hands to detect
MEDIAPIPE_MODEL_PATH = "models/hand_landmarker.task"  # Updated path to model file

# Screen mapping
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Feature extraction thresholds
PINCH_DISTANCE_THRESHOLD = 0.3  # Normalized (0-1), where 1 = fully spread

# Temporal analysis
TEMPORAL_WINDOW_SIZE = 15  # Number of frames in sliding window
TEMPORAL_FPS = 30

# Temporal thresholds
VELOCITY_THRESHOLD = 0.01  # Movement detection
STABILITY_THRESHOLD = 0.7  # Palm stability score
PINCH_THRESHOLD = 0.3  # Thumb-index distance for pinch detection
FINGER_MOTION_THRESHOLD = 0.02  # Finger movement magnitude

# State management timings (in seconds)
CLICK_HOLD_TIME = 0.15  # 150 ms minimum hold for click
COOLDOWN_TIME = 0.2  # 200 ms cooldown after discrete action
DRAG_MIN_TIME = 0.1  # 100 ms minimum drag duration

# Action execution smoothing
CURSOR_SENSITIVITY = 2.5  # Amplification factor: 1.0 = normal, 2.0 = 2x sensitivity, 3.0 = 3x sensitivity
CURSOR_SMOOTHING_FACTOR = 0.6  # 0 = no smoothing, 1 = full smoothing
MOUSE_PAUSE = 0.01  # PyAutoGUI pause between commands (in seconds)

# Debug visualization
DEBUG_MODE_DEFAULT = True
MOUSE_ENABLED_DEFAULT = False  # Safety: don't move mouse on startup
SHOW_LANDMARKS = True
SHOW_STATE_INFO = True
SHOW_MOTION_INDICATORS = True
SHOW_FINGER_STATES = True

# Performance
TARGET_FPS = 30
ENABLE_FPS_COUNTER = True

# Custom model thresholds
CUSTOM_STAGE_ONE_SCORE_THRESHOLD = 0.12
