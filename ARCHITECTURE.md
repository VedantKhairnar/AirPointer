# AirPointer: System Architecture & Design Documentation

## Table of Contents
1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Module Details](#module-details)
4. [Data Flow](#data-flow)
5. [State Machine](#state-machine)
6. [Temporal Logic](#temporal-logic)
7. [Design Rationale](#design-rationale)

---

## Overview

AirPointer is a **touchless mouse/trackpad replacement** that interprets hand motion in free space as continuous input device. Unlike gesture recognition systems, AirPointer:

- **Does NOT classify discrete gestures**
- **Does NOT require training or per-user calibration**
- **Infers intent from hand pose, motion, and time**
- **Adapts naturally to user intent**
- **Prioritizes correctness over sensitivity**

### Philosophy

> If it's not clearly intentional, don't do it.

The system is conservative: it prefers silence to false positives. Better to miss an action than trigger the wrong one.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VIDEO STREAM (30 FPS)                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
          ┌──────────────────────────────────┐
          │  PERCEPTION LAYER                │
          │ (MediaPipe Hands + Camera I/O)   │
          │ Outputs: 21 landmarks [x,y,z]   │
          └──────────────────────┬───────────┘
                            │
                            ↓
          ┌──────────────────────────────────┐
          │  FEATURE EXTRACTION              │
          │ Landmark → Physical Features     │
          │ Palm position, velocity, angles  │
          │ Finger states, pinch distance    │
          └──────────────────────┬───────────┘
                            │
                            ↓
          ┌──────────────────────────────────┐
          │  TEMPORAL ANALYSIS               │
          │ 15-frame sliding window          │
          │ Motion trends, stability scores  │
          │ State inference signals          │
          └──────────────────────┬───────────┘
                            │
                            ↓
          ┌──────────────────────────────────┐
          │  STATE MANAGEMENT (FSM)          │
          │ IDLE → MOVE → CLICK → DRAG       │
          │        → SCROLL → COOLDOWN       │
          │ Priority arbitration             │
          └──────────────────────┬───────────┘
                            │
                            ↓
          ┌──────────────────────────────────┐
          │  ACTION EXECUTION                │
          │ OS-level mouse/trackpad control  │
          │ pyautogui: move, click, drag     │
          └──────────────────────┬───────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM MOUSE EVENT                       │
│                    (OS Cursor, Click, Drag)                │
└─────────────────────────────────────────────────────────────┘
          ↑
          │ (Parallel path, no latency)
          │
          ┌──────────────────────────────────┐
          │  DEBUG VISUALIZATION             │
          │ State overlay, motion indicators │
          │ Landmark visualization           │
          │ Real-time feedback to user       │
          └──────────────────────────────────┘
```

---

## Module Details

### 1. Perception Layer (`perception/`)

**Responsibility**: Extract hand geometry from camera frames.

#### Components
- `CameraHandler`: OpenCV-based camera input with frame capture
- `HandDetector`: MediaPipe Hands integration

#### Key Design Decisions

**Why MediaPipe Hands?**
- Pretrained, no fine-tuning required
- 21-point hand model (standard in industry)
- Real-time, CPU-efficient
- Robust across hand sizes, lighting, rotation
- No user calibration needed

**What it outputs:**
- 21 landmarks in normalized coordinates [x ∈ [0,1], y ∈ [0,1], z (depth)]
- Detection confidence score

#### No Gesture Classification Here
The detector outputs **pure hand geometry only**—it does NOT interpret what the hand is doing. This interpretation happens downstream in feature extraction and temporal analysis.

---

### 2. Feature Extraction (`feature_extraction/`)

**Responsibility**: Convert 21 landmarks into interpretable physical features.

#### Extracted Features

| Feature | Definition | Use |
|---------|-----------|-----|
| **Palm Position** | Wrist coordinate [x, y, z] | Cursor movement mapping |
| **Palm Velocity** | Euclidean distance moved per frame | Movement detection |
| **Palm Acceleration** | Change in velocity | Stability detection |
| **Palm Stability** | 1 / (1 + acceleration × 10) | Click qualification |
| **Finger Extension** | Tip distance > MCP × 0.8 | Gesture detection |
| **Pinch Distance** | Thumb-Index distance (normalized) | Drag/pinch detection |
| **Finger Spread** | Std dev of finger positions | Overall hand openness |

#### Implementation Details

- **Finger Extension Detection**: Compare distance from finger tip to MCP (metacarpal joint). Extended if tip is sufficiently far from base.
- **Pinch Distance**: Euclidean distance normalized by typical spread (~0.3). Range: 0 (pinched) to 1 (spread).
- **Stability Scoring**: Exponential decay: `1 / (1 + accel × 10)`. Penalizes high acceleration sharply.

#### Why This Approach?

Each feature represents a **measurable physical quantity**, not an abstract gesture. This makes the system:
- Explainable
- Debuggable
- Adjustable without retraining
- Robust to variability

---

### 3. Temporal Analysis (`temporal_analysis/`)

**Responsibility**: Analyze motion patterns over time to infer intent.

#### Sliding Window
- **Size**: 15 frames ≈ 500ms @ 30 FPS
- **Why 15 frames?**
  - Sufficient to smooth jitter
  - Still responsive to intent changes
  - Captures ~half a second of motion

#### Temporal Statistics

For each frame's features, compute:
1. **Trends** (averages over window)
   - `palm_velocity_trend` = mean(velocities)
   - `palm_acceleration_trend` = mean(accelerations)
   - `palm_stability_trend` = mean(stability_scores)

2. **State Indicators** (derived from trends)
   - `is_moving`: velocity_trend > VELOCITY_THRESHOLD
   - `is_stable`: stability_trend > STABILITY_THRESHOLD
   - `is_transitioning`: moving but unstable

3. **Engagement Tracking**
   - `is_pinched`: mean(pinch_distances) < PINCH_THRESHOLD
   - `pinch_duration`: time pinch has been held

4. **Confidence**
   - `confidence`: min(1.0, history_length / window_size)

#### Why Not Frame-by-Frame?

A single frame is **noise**. The same visual pattern repeated for 500ms is **intent**.

Example:
- Frame 1: Finger tremor looks like "index extension" → WRONG
- Frames 1-15: Index steadily extends → RIGHT

---

### 4. State Management (`state_management/`)

**Responsibility**: Finite State Machine that infers interaction intent.

#### States

```
                       ┌─────────────┐
                       │    IDLE     │
                       └──────┬──────┘
                              │
                  ┌───────────┼───────────┐
                  ↓           ↓           ↓
             ┌────────┐  ┌─────────┐  ┌─────────┐
             │  MOVE  │  │  SCROLL │  │  CLICK  │
             └────────┘  └─────────┘  │ ENGAGED │
                              ↑        └────┬────┘
                              │             │
                              │    ┌────────↓────────┐
                              │    ↓                 │
                         ┌──────────────┐    ┌──────────┐
                         │     DRAG     │    │ COOLDOWN │
                         └──────────────┘    └──────────┘
                              │                    │
                              └────────┬───────────┘
                                       ↓
                                   IDLE (reset)
```

#### State Transitions & Logic

**IDLE → MOVE**
- Condition: `is_moving AND NOT pinched AND low_finger_motion`
- Output: Continuous cursor movement

**IDLE → CLICK_ENGAGED**
- Condition: `palm_stable AND deliberate_finger_motion`
- Temporal: Must hold for ≥150ms (CLICK_HOLD_TIME)
- Output: Single mouse click

**IDLE → DRAG**
- Condition: `pinch_sustained (>100ms) AND palm_moving`
- Output: Mouse press down, move cursor, eventually mouse up

**IDLE → SCROLL**
- Condition: `palm_moving AND NOT pinched AND vertical_motion`
- Output: Scroll events in detected direction

**Any → COOLDOWN**
- After discrete action (CLICK, SCROLL)
- Duration: 200ms (COOLDOWN_TIME)
- Purpose: Prevent double-fires and accidental chaining

#### Priority Arbitration

When multiple conditions could be true:
1. **DRAG** (highest) – pinch + move overrides everything
2. **CLICK** – intentional finger action
3. **SCROLL** – vertical motion
4. **MOVE** – continuous smooth movement
5. **IDLE** (lowest) – default

**Example**: If user is dragging and accidentally clicks, DRAG wins.

#### Temporal Confirmation

No action fires immediately. Discrete actions require:
- **Condition hold time**: Action condition must be true for min duration
- **Stability threshold**: Multiple frames must agree on state

This prevents:
- Accidental clicks from finger tremor
- Unintended state flips
- Double-clicks from jitter

---

### 5. Action Execution (`action_execution/`)

**Responsibility**: Map states to OS-level mouse/trackpad events.

#### Actions

| State | Action | Implementation |
|-------|--------|-----------------|
| **MOVE** | Cursor movement | `pyautogui.moveTo(x, y)` with smoothing |
| **CLICK_ENGAGED** | Left click | `pyautogui.click(button='left')` |
| **DRAG** | Drag operation | `mouseDown()` → `moveTo()` → `mouseUp()` |
| **SCROLL** | Scroll event | `pyautogui.scroll(amount)` |

#### Cursor Smoothing

Raw palm position → Smoothed cursor position:
```
smooth_x = prev_x × (1 - factor) + raw_x × factor
smooth_y = prev_y × (1 - factor) + raw_y × factor
```

**Smoothing factor** (default: 0.6):
- 0 = no smoothing (jittery)
- 1 = maximum smoothing (laggy)

**Benefit**: Reduces high-frequency jitter while maintaining responsiveness.

#### Screen Mapping

Camera frame [0, frame_width] → Screen [0, screen_width]
```
cursor_x = palm_x × (screen_width / frame_width)
```

Configurable in `config.py` for different screen resolutions.

---

### 6. Debug Visualization (`debug/`)

**Responsibility**: Render system state and debugging information.

#### Overlays

1. **State Box** (top-left)
   - Current state (color-coded)
   - State duration
   - Last action triggered

2. **Motion Indicators** (top-right)
   - Velocity trend
   - Stability score
   - Pinch status and distance
   - Movement direction

3. **Finger States** (bottom-left)
   - Per-finger extension state (Thumb, Index, Middle, Ring, Pinky)
   - Color: Green = extended, Red = bent

4. **Palm Indicator**
   - Circle at palm center
   - Ring showing stability
   - Velocity vector (arrow)

5. **Performance Metrics**
   - FPS counter
   - Mouse control status
   - Confidence score

#### Color Scheme

- **Green**: Extended fingers, moving, stable
- **Red**: Bent fingers, engaged, alert
- **Yellow**: Transitional, motion detected
- **Cyan**: Palm center, drag engaged
- **Blue**: IDLE state
- **Gray**: COOLDOWN state

---

## Data Flow

### Per-Frame Pipeline (30 FPS ≈ 33ms per frame)

```
Frame N arrives (timestamp T)
  │
  ├→ Perception: Hand detection (5-10ms)
  │    └→ landmarks[21, 3]
  │
  ├→ Feature Extraction (1-2ms)
  │    └→ HandFeatures (palm, velocity, finger states, etc.)
  │
  ├→ Temporal Analysis (1-2ms)
  │    └→ TemporalFeatures (trends, is_moving, is_pinched, etc.)
  │
  ├→ State Management (1ms)
  │    └→ StateInfo (current_state, pending_action)
  │
  ├→ Action Execution (0.5-1ms, if state changed)
  │    └→ OS mouse event
  │
  └→ Debug Visualization (3-5ms, if debug_mode=True)
       └→ Rendered frame with overlays

Total latency: ~50-100ms (camera → screen response)
```

### Memory Footprint

- Landmark history: 21 points × 15 frames × 4 bytes × 3 coords ≈ 4 KB
- Feature history: ~500 bytes
- State machine: ~1 KB
- Total: ~2-3 MB for all module data

---

## State Machine Details

### State Transition Conditions (in Code)

```python
# Pseudocode for state transitions

if temporal_features.is_pinched and temporal_features.pinch_duration > 0.1:
    # DRAG: pinch sustained + movement
    if temporal_features.palm_velocity_trend > 0.008:
        state = DRAG
        action = update_drag_position()

elif temporal_features.palm_stability_trend > 0.75:
    # CLICK: palm stable + deliberate finger motion
    if 0.015 < finger_motion < 0.05:
        if current_time - click_start_time >= 0.15:  # 150ms hold
            state = CLICK_ENGAGED
            action = click()
            enter_cooldown()

elif temporal_features.palm_velocity_trend > 0.01:
    # SCROLL or MOVE based on direction
    if detect_vertical_motion():
        state = SCROLL
        action = scroll(direction, amount)
    else:
        state = MOVE
        action = move_cursor()

else:
    state = IDLE
```

### Cooldown Mechanism

After discrete action (CLICK, SCROLL):
1. Enter COOLDOWN state
2. Set timer: `cooldown_until = now + 0.2 seconds`
3. Ignore all state transitions until timer expires
4. Return to IDLE when timer expires

**Purpose**: Prevent double-fires and forced state separation.

---

## Temporal Logic Deep Dive

### Why Temporal Analysis is Critical

Consider this scenario:
```
Frame 1: Index finger starts to extend (0.1m distance from MCP)
Frame 2-15: Jitter (tremor, breathing) makes distance fluctuate 0.1m ± 0.05m

Option A: Frame-by-frame detection
  → Multiple false "click" signals, double-click detected

Option B: 500ms sliding window
  → Average distance = 0.1m ± 0.03m (smoothed)
  → Detect consistent extension
  → Single, confident click
```

**Temporal features act as a low-pass filter, removing noise while preserving intent.**

### Motion Consistency

The system tracks not just "is velocity high?" but "is velocity consistently high?"

```
is_moving = (mean_velocity_over_window > THRESHOLD)
            AND (std_dev_velocity < acceptable_jitter)
```

This prevents:
- Accidental spasms triggering movement
- Quick flicks being interpreted as sustained motion

### Pinch Tracking

Pinch detection isn't just "distance < threshold." It's:
```
is_pinched = (current_distance < 0.3)
            AND (pinch_sustained_for > 0.1 seconds)

pinch_duration = sum of all frames where pinched within window
```

This ensures:
- Brief finger crossing doesn't trigger drag
- User must intentionally pinch for 100ms
- Drag only starts after confirmed intent

---

## Design Rationale

### 1. No Machine Learning

**Why?**
- Gesture classification requires labeled training data
- Per-user variation requires per-user models
- Ambiguous cases require decision logic anyway
- Hand mechanics are well-understood physics

**Consequence**: Pure rule-based + temporal analysis system is more:
- Explainable (no black box)
- Debuggable (trace every decision)
- Configurable (tune thresholds without retraining)
- Fair (same behavior for all users)

### 2. Implicit Gesture Emergence

**Gesture = change pattern over time**, not static pose.

Instead of classifying:
> "This pose is a 'click' gesture"

We infer:
> "This hand has been stable for 150ms, then suddenly the index extended. This matches a click pattern."

**Benefit**: Users don't need to learn "click pose." They naturally perform actions.

### 3. Priority Arbitration

**Why not parallel actions?**
- Can't click AND drag simultaneously (mutually exclusive states)
- Can't move AND click (palm must be stable for click)
- Can't scroll AND drag (both conflict for pinch input)

**Priority scheme**:
- Resolves conflicts deterministically
- Prevents accidental action combinations
- Gives drag highest priority (most intentional, requires sustained action)

### 4. Temporal Confirmation

**Why minimum hold times?**
- Human intent takes time to manifest
- Jitter is fast (<50ms), intent is slow (>150ms)
- Separates accidental tremor from deliberate action

**Why cooldown after actions?**
- Prevents state machine from re-triggering
- Allows user to transition hand position
- Reduces unintended action cascades

### 5. Stability Scoring

Instead of binary "moving / not moving":
```
stability = 1 / (1 + acceleration × 10)
```

Exponential curve:
- Small acceleration → high stability
- High acceleration → near-zero stability

**Benefit**: Smooth gradient, no hard thresholds. Can detect "almost stable" vs. "very stable."

---

## Failure Modes & Mitigations

| Failure | Cause | Mitigation |
|---------|-------|-----------|
| **Double-click** | Jitter on action | Temporal confirmation (150ms) + cooldown (200ms) |
| **Accidental drag** | Brief pinch | Require sustained pinch (>100ms) |
| **Click during move** | Overlapping conditions | Priority: DRAG > CLICK > MOVE |
| **Tremor → movement** | Hand instability | Stability scoring filters jitter |
| **Hand enters/exits frame** | Rapid detection loss | Temporal window smooths transitions |
| **Conflicting actions** | Ambiguous input | Arbitration rule (priority scheme) |

---

## Summary

AirPointer is a **modular, rule-based, temporal-aware input system** that:

1. ✓ Extracts hand geometry (MediaPipe, no training)
2. ✓ Converts to physical features (interpretable)
3. ✓ Analyzes motion over time (sliding window)
4. ✓ Infers intent via FSM (priority-based)
5. ✓ Executes actions (OS-level mouse)
6. ✓ Visualizes state (debugging + feedback)

**Design Philosophy**: When in doubt, do nothing. Better to miss an action than trigger the wrong one.

**Result**: A touchless input system that feels like a real mouse, not a gesture recognizer.
