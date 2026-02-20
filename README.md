# AirPointer: Touchless Mouse/Trackpad Replacement

A production-grade touchless input system that replaces traditional mouse + laptop trackpad using hand motion in free space.

## Overview

AirPointer interprets hand gestures as continuous input, exactly like a mouse or trackpad held in the air. The system automatically infers intent (move, click, drag, scroll) from hand pose, motion, and time—without explicit gesture modes or user training.

### Design Philosophy

- **No gesture classification**: The system treats the hand as a continuous input device
- **Implicit gesture emergence**: Actions arise naturally from hand mechanics over time
- **UI/UX correctness**: Prioritizes doing nothing over doing the wrong action
- **Mouse/trackpad familiarity**: Users feel like they're using a real input device, not performing gestures    

---

## Architecture

The pipeline consists of **six modular components**:

```
Camera Input 
    ↓
Perception (MediaPipe Hands)
    ↓
Feature Extraction (physical features)
    ↓
Temporal Analysis (sliding window, motion patterns)
    ↓
State Management (FSM for interaction intent)
    ↓
Action Execution (OS-level mouse events)
    ↓
Debug Visualization
```

### Module Breakdown

1. **Perception** (`perception/`)
   - Camera input handling
   - MediaPipe Tasks API for 21-landmark extraction
   - Real-time, CPU-efficient hand tracking using HandLandmarker

2. **Feature Extraction** (`feature_extraction/`)
   - Converts 21 landmarks → interpretable physical features
   - Palm position, velocity, acceleration, stability
   - Finger extension states, pinch distance, finger spread

3. **Temporal Analysis** (`temporal_analysis/`)
   - Maintains 10-20 frame sliding window
   - Computes motion deltas and trends
   - Detects stable vs. moving states
   - Tracks pinch duration

4. **State Management** (`state_management/`)
   - Finite State Machine with states: IDLE, MOVE, CLICK_ENGAGED, DRAG, SCROLL, COOLDOWN
   - Priority-based arbitration (DRAG > CLICK > SCROLL > MOVE)
   - Temporal confirmation (150ms for click, 100ms for drag)
   - Cooldown after discrete actions (200ms)

5. **Action Execution** (`action_execution/`)
   - OS-level mouse control (pyautogui)
   - Cursor movement with smoothing
   - Click, double-click, drag, scroll
   - Adaptive screen mapping

6. **Debug Visualization** (`debug/`)
   - Real-time state overlay
   - Landmark visualization
   - Motion indicators, stability feedback
   - FPS counter, action logging

---

## Installation

### Prerequisites
- Python 3.8+ (tested on Python 3.10-3.14)
- Webcam (integrated or USB)
- macOS / Linux / Windows

### Setup

```bash
# 1. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download MediaPipe hand landmark model
curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o hand_landmarker.task

# Note: If curl fails, manually download the 7.5 MB hand_landmarker.task file
# from the MediaPipe model repository and place it in the project root directory
```

**Dependencies:**
- `mediapipe>=0.10.0`: Hand tracking using MediaPipe Tasks API
- `opencv-python>=4.8.0`: Camera input and visualization
- `numpy>=1.24.0`: Numerical computations
- `pyautogui>=0.9.53`: OS-level mouse control

**Important:** The `hand_landmarker.task` model file (7.5 MB) is required for hand detection. The system uses MediaPipe's Tasks API for real-time hand landmark detection.

### Automated Setup (Recommended)

For convenience, use the provided setup script:

```bash
chmod +x start.sh
./start.sh
```

This script will:
1. Check Python version
2. Create/activate virtual environment
3. Install all dependencies
4. Download the hand landmark model automatically
5. Run tests to validate the system
6. Start AirPointer

---

## Usage

### Quick Start

```bash
# Option 1: Using the automated script
./start.sh

# Option 2: Manual start
python main.py
```

### Keyboard Controls

While the system is running:
- **`q`**: Quit AirPointer
- **`d`**: Toggle debug mode (visual overlay)
- **`m`**: Enable/disable mouse control (safety switch)

### Debug Mode

Debug mode displays:
- Current interaction state (IDLE, MOVE, CLICK_ENGAGED, DRAG, SCROLL, COOLDOWN)
- State duration
- Last action triggered
- Palm position and motion indicators
- Finger extension states (Thumb, Index, Middle, Ring, Pinky)
- Pinch status and distance
- Confidence and stability scores
- FPS counter

---

## Interaction Model

### Cursor Movement (MOVE state)
- **Condition**: Palm moving smoothly, no finger engagement
- **Output**: Proportional cursor movement
- **Feel**: Like holding a mouse in the air

### Click (CLICK_ENGAGED state)
- **Condition**: Palm stable, deliberate index finger motion (no pinch)
- **Duration**: Must hold for ≥150ms to prevent accidental clicks
- **Output**: Left mouse click
- **Feel**: Natural, finger-based clicking

### Drag (DRAG state)
- **Condition**: Thumb-index pinch sustained, palm moving
- **Duration**: ≥100ms pinch duration before drag initiates
- **Output**: Left-button drag while pinched, release ends drag
- **Feel**: Familiar pinch-and-drag from trackpads

### Scroll (SCROLL state)
- **Condition**: Palm moving vertically, no pinch
- **Output**: Proportional scroll in detected direction
- **Feel**: Like swiping on a trackpad

### Idle (IDLE state)
- **Condition**: No significant motion or engagement
- **Output**: No action
- **Feel**: Resting; no accidental commands

---

## Key Design Decisions

### 1. No Gesture Classifier
Instead of training a model to classify discrete gestures, the system uses **rule-based temporal logic**. Actions emerge from:
- Palm velocity and acceleration
- Finger extension patterns
- Pinch distance and duration
- Motion consistency over time

### 2. Sliding Time Window (15 frames ≈ 500ms)
Decisions are never frame-to-frame. All temporal features are averaged or computed across a sliding window to:
- Suppress jitter and noise
- Detect true intent (not accidental motion)
- Enable smooth, predictable behavior

### 3. Priority Arbitration
When multiple actions are possible:
1. **DRAG** (highest priority) – overrides everything
2. **CLICK** – intentional finger action
3. **SCROLL** – vertical motion
4. **MOVE** – continuous palm movement
5. **IDLE** – default

This prevents conflicting actions (e.g., clicking during drag).

### 4. Temporal Confirmation & Cooldown
Every discrete action must:
- Meet its condition for a minimum duration (e.g., 150ms for click)
- Be followed by a cooldown (200ms) before the next action

This prevents:
- Accidental double-clicks
- Rapid unintended state changes
- Left+right click collisions

### 5. Stability Over Cleverness
When in doubt, the system does nothing. This philosophy ensures:
- No false positives
- User never feels surprised
- Behavior is predictable and safe

---

## Customization

### Adjusting Sensitivity

Edit thresholds in respective modules:

**Feature Extraction** (`feature_extraction/extractor.py`):
- `thumb_index_distance / 0.3` – Pinch sensitivity

**Temporal Analysis** (`temporal_analysis/analyzer.py`):
- `VELOCITY_THRESHOLD` – Movement sensitivity
- `STABILITY_THRESHOLD` – Steadiness requirement
- `PINCH_THRESHOLD` – Pinch distance threshold

**State Management** (`state_management/state_machine.py`):
- `CLICK_HOLD_TIME` – How long to hold for a click (default: 150ms)
- `COOLDOWN_TIME` – Pause between actions (default: 200ms)
- `DRAG_MIN_TIME` – Minimum drag duration (default: 100ms)

**Action Execution** (`action_execution/executor.py`):
- `smoothing_factor` – Cursor smoothing (0-1, higher = smoother)
- `scale_x`, `scale_y` – Screen mapping calibration

### Screen Calibration

If cursor movement doesn't map correctly:
1. Measure your screen resolution (System Preferences → Displays)
2. Edit `ActionExecutor` initialization in `main.py`:
   ```python
   action_executor = ActionExecutor(
       screen_width=YOUR_WIDTH,   # e.g., 2560
       screen_height=YOUR_HEIGHT,  # e.g., 1440
       frame_width=1280,
       frame_height=720
   )
   ```

---

## Performance

- **CPU Usage**: ~15-25% on modern macOS (M1/M2)
- **Latency**: ~50-100ms (camera → cursor)
- **Frame Rate**: 25-30 FPS
- **Memory**: ~150-200MB

Tested on:
- macOS (13.x, 14.x)
- Intel and Apple Silicon

---

## Limitations & Future Work

### Current Limitations
- Single-hand tracking only
- No multi-touch gestures (two-hand interactions)
- Requires good lighting for MediaPipe
- Limited to left-mouse operations (no right-click calibration yet)

### Future Enhancements
1. **Right-click**: Add pinch-with-middle-finger for right-click
2. **Velocity-based cursor acceleration**: Smooth acceleration for large movements
3. **Temporal transformer (optional)**: Use transformer to model intent for ambiguous cases
4. **Hand pose presets**: Quick access to system shortcuts (thumbs-up for mute, etc.)
5. **Adaptive thresholds**: Auto-adjust based on user's baseline hand size/movement
6. **Multi-hand support**: Two-hand gestures for resize, rotate, etc.

---

## Troubleshooting

### Model Not Found / Hand Detection Fails
- Ensure `hand_landmarker.task` file exists in the project root directory
- File should be approximately 7.5 MB in size (not 236 bytes)
- Download manually if needed:
  ```bash
  curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o hand_landmarker.task
  ```
- Verify the file is not corrupted: `ls -lh hand_landmarker.task`

### MediaPipe Import Errors
- Ensure mediapipe>=0.10.0 is installed: `pip show mediapipe`
- The system uses MediaPipe Tasks API (not the legacy solutions API)
- Reinstall if needed: `pip uninstall mediapipe && pip install mediapipe`

### Cursor Not Moving
- Enable mouse control: Press `m` during runtime
- Ensure hand is visible in camera frame
- Check screen resolution in `ActionExecutor`

### Clicks Firing Too Easily
- Increase `CLICK_HOLD_TIME` in state management
- Increase `palm_stability` threshold

### Clicks Not Firing
- Ensure fingers are moving deliberately (not micro-tremors)
- Check debug mode to see if state reaches `CLICK_ENGAGED`

### Erratic Cursor Behavior
- Increase `smoothing_factor` in `ActionExecutor`
- Ensure stable lighting for hand detection
- Check MediaPipe confidence in debug output

### Performance Issues
- Run in release mode (disable debug visualization)
- Reduce frame resolution (1280x720 → 640x480)
- Close other applications using the webcam

---

## Code Structure

```
AirPointer/
├── main.py                          # Entry point, pipeline orchestration
├── main_advanced.py                 # Config-driven version
├── config.py                        # Centralized configuration
├── requirements.txt                 # Python dependencies
├── hand_landmarker.task            # MediaPipe hand landmark model (7.5 MB)
├── start.sh                         # Automated setup script
├── README.md                        # This file
├── perception/
│   ├── __init__.py
│   ├── camera_handler.py           # Camera I/O
│   └── hand_detector.py            # MediaPipe Tasks API integration
├── feature_extraction/
│   ├── __init__.py
│   └── extractor.py                # Landmark → features
├── temporal_analysis/
│   ├── __init__.py
│   └── analyzer.py                 # Sliding window, motion patterns
├── state_management/
│   ├── __init__.py
│   └── state_machine.py            # FSM, state transitions
├── action_execution/
│   ├── __init__.py
│   └── executor.py                 # OS mouse control
└── debug/
    ├── __init__.py
    └── visualizer.py               # Visualization & logging
```

---

## Citation

If you use AirPointer in research or projects, please cite:

```
AirPointer: Touchless Mouse/Trackpad Replacement
Computer Vision + HCI System Design
https://github.com/[your-repo]
```

---

## License

[Your License Here]

---

## Contact

For issues, suggestions, or contributions, please open an issue or contact the development team.

---

**Last Updated**: February 2026  
**Status**: Production-Ready Prototype
