# AirPointer: Troubleshooting & Advanced Tuning Guide

## Quick Diagnosis

Run the test script to validate all components:
```bash
python test.py
```

If tests pass, proceed to usage. If any test fails, see corresponding section below.

---

## Common Issues & Solutions

### Issue: "No hand detected" continuously

**Symptom**: Debug window shows "No hand detected" even with hand in frame

**Causes & Solutions**:

1. **Poor lighting**
   - MediaPipe relies on visual cues
   - Ensure adequate lighting (natural daylight or bright lamp)
   - Avoid backlighting (hand becomes silhouette)

2. **Hand not fully visible**
   - MediaPipe needs most of hand in frame
   - Keep hand centered, not at edges
   - Avoid partial hand (cut off by frame)

3. **Camera resolution mismatch**
   - Check config.py: `CAMERA_WIDTH`, `CAMERA_HEIGHT`
   - Verify camera supports chosen resolution
   - Try default: 1280x720

4. **MediaPipe model not found**
   - Ensure `hand_landmarker.task` file exists in project root
   - File should be approximately 7.5 MB
   - Download with:
     ```bash
     curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o hand_landmarker.task
     ```

5. **MediaPipe initialization fails**
   - AirPointer uses MediaPipe Tasks API (not the legacy solutions API)
   - Ensure mediapipe>=0.10.0 is installed
   - Verify hand_landmarker.task model is valid (>1 MB file size)
   - Check Python version compatibility (tested on 3.10-3.14)

### Issue: Cursor doesn't move

**Symptom**: Hand moves but cursor stays still

**Checks**:

1. **Mouse control disabled?**
   - During runtime, press `m` to enable mouse control
   - Watch console: should print "ENABLED"

2. **Wrong screen resolution**
   - Get actual screen resolution:
     - macOS: System Preferences → Displays
     - Linux: `xrandr`
     - Windows: Settings → Display
   - Update config.py:
     ```python
     SCREEN_WIDTH = 2560  # Your actual width
     SCREEN_HEIGHT = 1440  # Your actual height
     ```

3. **Camera mapping inverted**
   - Hand on left side of camera frame?
   - Press `d` to toggle debug mode
   - Check if palm circle position matches your hand
   - If inverted, camera might need flip adjustment in camera_handler.py:
     ```python
     # In get_frame():
     # frame = cv2.flip(frame, 1)  # Change flip code
     ```

4. **Frame resolution mismatch**
   - Config expects 1280x720
   - If camera outputs different resolution:
     ```python
     action_executor = ActionExecutor(
         frame_width=YOUR_ACTUAL_WIDTH,
         frame_height=YOUR_ACTUAL_HEIGHT,
         ...
     )
     ```

### Issue: Clicks fire too easily or too hard

**Symptom**: Accidental clicks while moving / Can't trigger clicks

**Too Easy** (accidental clicks):
1. Increase hold time:
   ```python
   # config.py
   CLICK_HOLD_TIME = 0.25  # Increase from 0.15
   ```

2. Increase stability requirement:
   ```python
   # state_machine.py, in _should_click():
   palm_stable = temporal_features.palm_stability_trend > 0.85  # from 0.75
   ```

3. Reduce finger motion sensitivity:
   ```python
   # state_machine.py, in _should_click():
   has_deliberate_motion = 0.02 < finger_motion < 0.04  # from 0.015-0.05
   ```

**Too Hard** (can't trigger clicks):
1. Decrease hold time:
   ```python
   CLICK_HOLD_TIME = 0.1  # Decrease from 0.15
   ```

2. Lower stability requirement:
   ```python
   palm_stable = temporal_features.palm_stability_trend > 0.65  # from 0.75
   ```

3. Increase finger motion window:
   ```python
   has_deliberate_motion = 0.01 < finger_motion < 0.06  # Wider range
   ```

### Issue: Erratic cursor movement

**Symptom**: Cursor jumps around even when hand is still

**Causes & Solutions**:

1. **Increase cursor smoothing**
   ```python
   # config.py
   CURSOR_SMOOTHING_FACTOR = 0.8  # Increase from 0.6
   ```
   Higher = smoother but more lag

2. **Reduce movement velocity threshold**
   ```python
   # config.py
   VELOCITY_THRESHOLD = 0.008  # More sensitive movement detection
   ```

3. **Check lighting**
   - Hand detection becomes unstable in poor light
   - MediaPipe trembles → cursor jitter

4. **Increase temporal window**
   ```python
   # config.py
   TEMPORAL_WINDOW_SIZE = 20  # from 15, more smoothing
   ```

### Issue: Drag won't initiate

**Symptom**: Pinch detected but drag doesn't start

**Checks**:

1. **Pinch threshold too strict**
   ```python
   # config.py
   PINCH_THRESHOLD = 0.35  # Increase (more lenient) from 0.3
   ```

2. **Minimum pinch duration too long**
   ```python
   # state_machine.py, in _should_drag():
   pinch_sustained = temporal_features.is_pinched and \
                     temporal_features.pinch_duration > 0.08  # from 0.1
   ```

3. **Finger distance calculation wrong**
   - Check feature extraction: is thumb-index distance being computed correctly?
   - Run test.py and check `thumb_index_distance` in output

### Issue: Scroll won't work

**Symptom**: Hand moves vertically but no scroll events

**Solutions**:

1. **Scroll detection is simplified**
   - Current code treats all motion as potential scroll
   - Edit state_machine.py to detect Y-axis motion specifically:
     ```python
     # Instead of generic palm_moving, check Y specifically:
     palm_y_velocity = (curr_y - prev_y) / dt
     is_vertical = abs(palm_y_velocity) > 0.015
     ```

2. **Lower velocity threshold**
   ```python
   # state_machine.py, in _should_scroll():
   palm_moving = temporal_features.palm_velocity_trend > 0.005  # from 0.01
   ```

3. **Check direction detection**
   - Currently, scroll always goes "down"
   - To detect direction:
     ```python
     # Modify temporal_analyzer.py to track Y-direction
     palm_y_velocity = (current_pos[1] - prev_pos[1])
     scroll_direction = 'down' if palm_y_velocity > 0 else 'up'
     ```

### Issue: Performance / Lag

**Symptom**: Low FPS, system feels sluggish

**Solutions** (in order of effectiveness):

1. **Disable debug visualization**
   - Press `d` to turn off debug mode
   - Visualization takes ~5ms per frame

2. **Reduce frame resolution**
   ```python
   # config.py
   CAMERA_WIDTH = 640  # from 1280
   CAMERA_HEIGHT = 480  # from 720
   ```

3. **Reduce temporal window**
   ```python
   # config.py
   TEMPORAL_WINDOW_SIZE = 10  # from 15
   ```

4. **Close other applications**
   - Webcam drivers, other video apps
   - Check Activity Monitor: look for camera-using processes

5. **Upgrade hardware**
   - System requires modern CPU
   - Older machines may struggle with MediaPipe

---

## Advanced Tuning

### Sensitivity Profile: Slow & Careful (Safe)

For users who want very deliberate input:

```python
# config.py

# More stable detection
CLICK_HOLD_TIME = 0.25
COOLDOWN_TIME = 0.3
DRAG_MIN_TIME = 0.15

# Less sensitive thresholds
VELOCITY_THRESHOLD = 0.015
STABILITY_THRESHOLD = 0.8
PINCH_THRESHOLD = 0.25  # Require stronger pinch

# More smoothing
CURSOR_SMOOTHING_FACTOR = 0.75

# Larger temporal window
TEMPORAL_WINDOW_SIZE = 20
```

### Sensitivity Profile: Fast & Responsive

For users who want quick reactions:

```python
# config.py

# Faster action confirmation
CLICK_HOLD_TIME = 0.1
COOLDOWN_TIME = 0.15
DRAG_MIN_TIME = 0.05

# More sensitive detection
VELOCITY_THRESHOLD = 0.005
STABILITY_THRESHOLD = 0.6
PINCH_THRESHOLD = 0.4  # Easier to trigger

# Less smoothing
CURSOR_SMOOTHING_FACTOR = 0.4

# Smaller temporal window
TEMPORAL_WINDOW_SIZE = 10
```

### Screen-Specific Calibration

For unusual resolutions or multi-monitor setups:

1. Measure your display resolution precisely
2. Update config.py:
   ```python
   SCREEN_WIDTH = 2560  # 5K iMac example
   SCREEN_HEIGHT = 1440
   ```

3. Test calibration:
   - Run main.py
   - Move hand to corner of camera frame
   - Observe where cursor appears on screen
   - Adjust SCREEN_WIDTH/HEIGHT if mapping is off

4. For multi-monitor:
   - Only one monitor supported currently
   - Choose monitor where you want cursor
   - Set SCREEN_WIDTH/HEIGHT to that monitor's resolution

### Hand Size Adaptation

Large hands vs. small hands can affect pinch distance.

**For large hands** (pinch too easy):
```python
PINCH_THRESHOLD = 0.25  # More stringent
```

**For small hands** (pinch too hard):
```python
PINCH_THRESHOLD = 0.4  # More lenient
```

To find your natural pinch distance:
1. Run main.py with debug mode on
2. Make a pinch gesture
3. Watch "Pinch: distance" value
4. Find comfortable threshold (usually 0.2-0.4)

### Finger Motion Tuning

Adjust how easily clicks are triggered based on finger movement.

**More deliberate click required**:
```python
# state_machine.py, _should_click():
has_deliberate_motion = 0.025 < finger_motion < 0.045  # Narrower range
```

**Easier/lighter click**:
```python
has_deliberate_motion = 0.01 < finger_motion < 0.06  # Wider range
```

---

## Performance Profiling

To measure which modules take the most time:

1. Add timing to main.py:
   ```python
   import time
   
   t1 = time.time()
   landmarks, _ = detector.detect(frame)
   print(f"Detection: {(time.time() - t1) * 1000:.1f}ms")
   
   t2 = time.time()
   features = extractor.extract_features(landmarks, t)
   print(f"Features: {(time.time() - t2) * 1000:.1f}ms")
   
   # ... repeat for other modules
   ```

2. Expected timings @ 30 FPS:
   - Detection (MediaPipe): 5-10ms
   - Features: 1-2ms
   - Temporal: 1-2ms
   - State: <1ms
   - Action: <1ms
   - Visualization: 3-5ms
   - **Total: ~15-25ms** (30 FPS headroom)

If any module exceeds expected time:
- Reduce resolution
- Simplify computation
- Check for bottlenecks

---

## Custom Gesture Implementation

To add a new action (e.g., "three-finger tap for right-click"):

### Step 1: Define feature in feature_extraction/extractor.py
```python
@dataclass
class HandFeatures:
    # ... existing features ...
    three_fingers_extended: bool  # NEW
```

### Step 2: Extract in extract_features()
```python
def extract_features(self, landmarks, timestamp):
    # ... existing code ...
    
    # Count extended fingers
    extended_count = sum([
        index_extended,
        middle_extended,
        ring_extended,
        # Exclude pinky for 3-finger detection
    ])
    three_fingers_extended = (extended_count == 3)
    
    return HandFeatures(
        # ... existing ...
        three_fingers_extended=three_fingers_extended
    )
```

### Step 3: Add state in state_management/state_machine.py
```python
class InteractionState(Enum):
    # ... existing ...
    RIGHT_CLICK = "RIGHT_CLICK"
```

### Step 4: Add transition logic
```python
def update(self, temporal_features):
    # ... priority 1, 2, 3, 4 ...
    
    # Priority 5: RIGHT_CLICK
    elif self._should_right_click(temporal_features):
        self._transition_to(InteractionState.RIGHT_CLICK, current_time)
        self.pending_action = "right_click"
    
    # ... rest of transitions ...

def _should_right_click(self, temporal_features):
    three_ext = temporal_features.current_features.three_fingers_extended
    stable = temporal_features.palm_stability_trend > 0.75
    return three_ext and stable
```

### Step 5: Add execution in action_execution/executor.py
```python
def execute_right_click(self):
    pyautogui.click(button='right')
```

### Step 6: Hook in main.py
```python
elif state_info.current_state == InteractionState.RIGHT_CLICK:
    if state_info.pending_action == "right_click" and mouse_enabled:
        action_executor.execute_right_click()
        state_manager.last_action = "RIGHT_CLICK"
        state_manager.enter_cooldown(time.time())
```

---

## Debugging Tips

### 1. Check Hand Detection First
- Run with debug mode on: `d`
- Do you see landmarks drawn? (green points + lines)
- If not, fix detection before debugging anything else

### 2. Watch Feature Extraction
- In debug overlay (top-right):
  - Is velocity trending correctly when you move?
  - Is stability high when you're still?
  - Is pinch distance changing when you pinch?

### 3. Verify State Transitions
- In debug overlay (top-left):
  - Does state match your intent?
  - Are you seeing CLICK_ENGAGED when you try to click?
  - Are you seeing DRAG when you pinch?

### 4. Isolate State Bugs
- Disable action execution temporarily:
  - Set `mouse_enabled = False` at startup
  - Observe state transitions without side effects
  - Determine if state machine is correct

### 5. Analyze Frame-by-Frame (Advanced)
```python
# Add to main.py to record frame data:
import json

frame_data = []

# During loop:
frame_data.append({
    'timestamp': current_timestamp,
    'palm_x': hand_features.palm_x,
    'palm_velocity': hand_features.palm_velocity,
    'state': state_info.current_state.value,
})

# After shutdown:
with open('analysis.json', 'w') as f:
    json.dump(frame_data, f)

# Analyze offline to find patterns
```

---

## Performance Optimization Checklist

- [ ] Disable debug mode in production
- [ ] Use optimized camera resolution (1280x720 default)
- [ ] Reduce temporal window if CPU-bound (15 → 10)
- [ ] Profile code with profiler to find bottlenecks
- [ ] Consider running on separate thread for UI responsiveness
- [ ] Cache MediaPipe model on startup
- [ ] Monitor memory usage for long sessions

---

## Questions?

If you encounter issues not covered here:

1. **Check debug output** (verbose logging)
2. **Run test.py** (component validation)
3. **Check ARCHITECTURE.md** (design documentation)
4. **Review config.py** (all tunable parameters)
5. **Inspect state transitions** (state_machine.py)

---

**Last Updated**: February 2026
