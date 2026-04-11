from typing import Optional

import numpy as np
import pyautogui

import config


class ActionExecutor:
    def __init__(self):
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = config.MOUSE_PAUSE

        self.screen_width = config.SCREEN_WIDTH
        self.screen_height = config.SCREEN_HEIGHT
        self.sensitivity = config.CURSOR_SENSITIVITY
        self.base_smoothing = config.CURSOR_SMOOTHING_FACTOR
        self.deadzone = 0.008  # Filters hand tremor during fingertip tracking
        self.max_step_px = 120.0  # Allows faster cursor response to quick hand movements
        self._stop_motion_threshold = 0.0022
        self._tail_snap_threshold = 0.006

        self._last_hand_pos: Optional[np.ndarray] = None
        self._filtered_hand_pos: Optional[np.ndarray] = None
        self._last_filtered_hand_pos: Optional[np.ndarray] = None
        self._cursor_pos: Optional[np.ndarray] = None
        self._dragging = False

    def execute_move(self, hand_features):
        # Use index finger tip (landmark 8) instead of palm for more direct, responsive control.
        index_tip = hand_features.landmarks[8, :2]  # x, y normalized [0, 1]
        current = np.array([index_tip[0], index_tip[1]], dtype=np.float32)
        hand_motion = 0.0
        if self._last_hand_pos is not None:
            hand_motion = float(np.linalg.norm(current - self._last_hand_pos))

        if self._filtered_hand_pos is None or self._last_filtered_hand_pos is None:
            self._filtered_hand_pos = current.copy()
            self._last_filtered_hand_pos = current.copy()
            self._last_hand_pos = current
            if self._cursor_pos is None:
                cursor_x, cursor_y = pyautogui.position()
                self._cursor_pos = np.array([cursor_x, cursor_y], dtype=np.float32)
            return

        # Smooth the hand position first, then move from the filtered trajectory.
        raw_delta = current - self._filtered_hand_pos
        delta_norm = float(np.linalg.norm(raw_delta))

        # For fingertip tracking: reduce smoothing baseline since tip is naturally more responsive.
        # Fingertip is noisier, so use tighter damping range to catch micro-movements.
        min_smoothing = float(np.clip(self.base_smoothing - 0.25, 0.30, 0.95))
        max_smoothing = float(np.clip(self.base_smoothing + 0.05, min_smoothing, 0.98))
        position_smoothing = float(np.clip(max_smoothing - delta_norm * 8.0, min_smoothing, max_smoothing))

        self._filtered_hand_pos = (
            position_smoothing * self._filtered_hand_pos + (1.0 - position_smoothing) * current
        )

        # If the hand is effectively stationary, snap filter state to prevent trailing cursor drift.
        if hand_motion < self._stop_motion_threshold:
            residual = float(np.linalg.norm(self._filtered_hand_pos - current))
            if residual < self._tail_snap_threshold:
                self._filtered_hand_pos = current.copy()
                self._last_filtered_hand_pos = current.copy()
                self._last_hand_pos = current
                return

        delta = self._filtered_hand_pos - self._last_filtered_hand_pos
        self._last_filtered_hand_pos = self._filtered_hand_pos.copy()
        self._last_hand_pos = current

        delta_norm = float(np.linalg.norm(delta))
        if delta_norm < self.deadzone:
            return

        dx = float(delta[0] * self.screen_width * self.sensitivity)
        dy = float(delta[1] * self.screen_height * self.sensitivity)

        step_norm = float(np.hypot(dx, dy))
        if step_norm > self.max_step_px and step_norm > 0.0:
            scale = self.max_step_px / step_norm
            dx *= scale
            dy *= scale

        try:
            if self._cursor_pos is None:
                current_x, current_y = pyautogui.position()
                self._cursor_pos = np.array([current_x, current_y], dtype=np.float32)

            self._cursor_pos[0] = float(np.clip(self._cursor_pos[0] + dx, 0, self.screen_width - 1))
            self._cursor_pos[1] = float(np.clip(self._cursor_pos[1] + dy, 0, self.screen_height - 1))
            new_x = int(self._cursor_pos[0])
            new_y = int(self._cursor_pos[1])
            pyautogui.moveTo(new_x, new_y)
        except Exception:
            return

    def reset_hand_tracking(self):
        if self._dragging:
            self.end_drag()
        self._last_hand_pos = None
        self._filtered_hand_pos = None
        self._last_filtered_hand_pos = None
        self._cursor_pos = None

    def execute_click(self, button: str = "left"):
        try:
            pyautogui.click(button=button)
        except Exception:
            return

    def start_drag(self, hand_features):
        if self._dragging:
            return
        self.execute_move(hand_features)
        try:
            pyautogui.mouseDown(button="left")
            self._dragging = True
        except Exception:
            self._dragging = False

    def update_drag(self, hand_features):
        if not self._dragging:
            return
        self.execute_move(hand_features)

    def end_drag(self):
        if not self._dragging:
            return
        try:
            pyautogui.mouseUp(button="left")
        finally:
            self._dragging = False

    def is_currently_dragging(self) -> bool:
        return self._dragging
