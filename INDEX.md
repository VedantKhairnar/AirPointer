# AirPointer: File Index & Quick Reference

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate system
python test.py

# 3. Run AirPointer
python main.py

# Controls: q=quit, d=debug toggle, m=mouse toggle
```

---

## File Structure

### Entry Points

| File | Purpose |
|------|---------|
| **main.py** | Standard entry point with full pipeline |
| **main_advanced.py** | Advanced version with class wrapper (recommended) |
| **test.py** | Component validation (run before main.py) |
| **start.sh** | Automated startup script |

### Configuration

| File | Purpose |
|------|---------|
| **config.py** | All tunable parameters (50+ settings) |
| **requirements.txt** | Python dependencies |

### Documentation

| File | Read This For | Length |
|------|--------|--------|
| **README.md** | User guide, features, installation | 350 lines |
| **ARCHITECTURE.md** | System design, module details, logic | 800+ lines |
| **TROUBLESHOOTING.md** | Debugging, tuning, custom gestures | 600+ lines |
| **IMPLEMENTATION.md** | Delivery summary, design decisions | 400+ lines |

### Modules

#### Perception (`perception/`)
Extract hand landmarks from camera

| File | Class | Purpose |
|------|-------|---------|
| camera_handler.py | `CameraHandler` | OpenCV camera input |
| hand_detector.py | `HandDetector` | MediaPipe Hands integration |

#### Feature Extraction (`feature_extraction/`)
Convert landmarks to physical features

| File | Class | Purpose |
|------|-------|---------|
| extractor.py | `FeatureExtractor` | Compute physical features |
| extractor.py | `HandFeatures` | Feature container (dataclass) |

#### Temporal Analysis (`temporal_analysis/`)
Analyze motion patterns over time

| File | Class | Purpose |
|------|-------|---------|
| analyzer.py | `TemporalAnalyzer` | Sliding window motion analysis |
| analyzer.py | `TemporalFeatures` | Temporal feature container |

#### State Management (`state_management/`)
Finite State Machine for intent

| File | Class | Purpose |
|------|-------|---------|
| state_machine.py | `StateManager` | FSM implementation |
| state_machine.py | `InteractionState` | State enum (IDLE, MOVE, etc.) |
| state_machine.py | `StateInfo` | State information container |

#### Action Execution (`action_execution/`)
OS-level mouse control

| File | Class | Purpose |
|------|-------|---------|
| executor.py | `ActionExecutor` | Mouse movement, click, drag, scroll |

#### Debug (`debug/`)
Real-time visualization

| File | Class | Purpose |
|------|-------|---------|
| visualizer.py | `DebugVisualizer` | State and motion overlays |

---

## Configuration Parameters

### Camera Settings (config.py)

```python
CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
```

### Screen Settings

```python
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
```

### Timing Parameters (seconds)

```python
CLICK_HOLD_TIME = 0.15        # Hold for click confirmation
COOLDOWN_TIME = 0.2            # Pause after discrete action
DRAG_MIN_TIME = 0.1            # Minimum drag duration
```

### Sensitivity Thresholds

```python
VELOCITY_THRESHOLD = 0.01      # Movement detection
STABILITY_THRESHOLD = 0.7      # Palm stability requirement
PINCH_THRESHOLD = 0.3          # Pinch distance threshold
FINGER_MOTION_THRESHOLD = 0.02 # Finger movement sensitivity
```

### Smoothing & Quality

```python
CURSOR_SMOOTHING_FACTOR = 0.6  # 0=none, 1=full
TEMPORAL_WINDOW_SIZE = 15      # Frames in sliding window
```

See **config.py** for all 50+ parameters.

---

## How the Pipeline Works

### Data Flow (per 33ms frame)

```
Frame arrives
    ↓
Perception: Hand detection (5-10ms)
    → 21 landmarks [x,y,z]
    ↓
Feature Extraction: Landmark analysis (1-2ms)
    → palm_x, palm_y, velocity, finger_states, pinch_distance
    ↓
Temporal Analysis: Motion patterns (1-2ms)
    → velocity_trend, stability_trend, is_pinched, pinch_duration
    ↓
State Management: FSM intent (1ms)
    → IDLE | MOVE | CLICK_ENGAGED | DRAG | SCROLL | COOLDOWN
    ↓
Action Execution: OS mouse event (0.5-1ms)
    → cursor.move(), click(), drag(), scroll()
    ↓
Debug Visualization: Overlay (3-5ms, if enabled)
    → render state, motion, landmarks

Total: ~15-25ms (under 33ms budget for 30 FPS)
```

---

## Key Design Concepts

### 1. No Gesture Classification
- No ML, no training, no gesture labels
- Pure rule-based temporal logic
- Actions inferred from hand mechanics

### 2. Temporal Window (15 frames)
- Single frame = noise
- 500ms of consistent motion = intent
- Filters jitter, detects real actions

### 3. State Machine
```
States: IDLE, MOVE, CLICK_ENGAGED, DRAG, SCROLL, COOLDOWN
Transitions: Explicit rules, no ambiguity
Priority: DRAG > CLICK > SCROLL > MOVE > IDLE
```

### 4. Temporal Confirmation
- Click: ≥150ms hold
- Drag: ≥100ms pinch duration
- Cooldown: 200ms between actions

### 5. Conservative Behavior
- If unsure, do nothing
- Prefer false negatives to false positives
- Users never surprised

---

## Common Tasks

### Enable/Disable Mouse Control
```python
# During runtime
press 'm' to toggle
```

### Toggle Debug Visualization
```python
# During runtime
press 'd' to toggle
```

### Adjust Click Sensitivity
```python
# In config.py
CLICK_HOLD_TIME = 0.1      # Easier (faster)
CLICK_HOLD_TIME = 0.25     # Harder (more deliberate)
```

### Adjust Cursor Smoothing
```python
# In config.py
CURSOR_SMOOTHING_FACTOR = 0.4  # Responsive (jittery)
CURSOR_SMOOTHING_FACTOR = 0.8  # Smooth (laggy)
```

### Add Custom Gesture
See **TROUBLESHOOTING.md** section "Custom Gesture Implementation"

### Calibrate Screen Resolution
```python
# In config.py
SCREEN_WIDTH = 2560   # Your actual screen width
SCREEN_HEIGHT = 1440  # Your actual screen height
```

---

## Troubleshooting Checklist

1. **No hand detected?**
   - Check lighting (needs good illumination)
   - Ensure hand fully visible in camera frame
   - Reduce MediaPipe confidence in config.py

2. **Cursor not moving?**
   - Press 'm' to enable mouse control
   - Check screen resolution in config.py
   - Verify hand detection (press 'd' for debug)

3. **Clicks firing too easily?**
   - Increase CLICK_HOLD_TIME in config.py
   - Increase STABILITY_THRESHOLD
   - Reduce FINGER_MOTION_THRESHOLD

4. **Erratic cursor movement?**
   - Increase CURSOR_SMOOTHING_FACTOR
   - Check lighting (detection instability)
   - Reduce VELOCITY_THRESHOLD

See **TROUBLESHOOTING.md** for detailed solutions.

---

## Performance Tips

1. **Disable debug visualization** (press 'd') for better FPS
2. **Reduce frame resolution** in config.py (1280x720 → 640x480)
3. **Reduce temporal window** (15 → 10 frames)
4. **Close other camera apps** that might conflict
5. **Monitor CPU usage** with Activity Monitor

---

## Testing

Run validation suite:
```bash
python test.py
```

Expected output:
```
✓ All imports successful
✓ Feature extraction working
✓ Temporal analysis working
✓ State management working
✓ Action execution initialized
✓ Debug visualization working

✓ All tests passed! AirPointer is ready to run.
```

---

## Documentation Map

```
README.md (START HERE)
├─ Overview
├─ Installation
├─ Usage
├─ Interaction Model
└─ Customization

ARCHITECTURE.md (TECHNICAL)
├─ Pipeline architecture
├─ Module details
├─ Data flow
├─ State machine logic
├─ Temporal analysis
└─ Design rationale

TROUBLESHOOTING.md (DEBUGGING)
├─ Common issues
├─ Advanced tuning
├─ Performance optimization
├─ Custom gestures
└─ Debug tips

config.py (CONFIGURATION)
└─ 50+ tunable parameters

test.py (VALIDATION)
└─ Component testing
```

---

## Quick Reference: States & Actions

| State | Trigger | Output |
|-------|---------|--------|
| **MOVE** | Palm moving, no pinch | Cursor movement |
| **CLICK_ENGAGED** | Palm stable, finger extension (≥150ms) | Left click |
| **DRAG** | Pinch sustained (≥100ms), palm moving | Mouse drag |
| **SCROLL** | Vertical motion, no pinch | Scroll event |
| **COOLDOWN** | After discrete action | 200ms pause |
| **IDLE** | No activity | Nothing |

---

## Support

- **User guide**: README.md
- **Technical details**: ARCHITECTURE.md
- **Debugging**: TROUBLESHOOTING.md
- **Implementation**: IMPLEMENTATION.md
- **Configuration**: config.py

---

**Version**: 1.0 Production Ready
**Last Updated**: February 2026
**Status**: ✓ Complete and tested
