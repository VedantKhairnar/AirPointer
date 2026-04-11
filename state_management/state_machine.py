from dataclasses import dataclass
from enum import Enum, auto

import config


class InteractionState(Enum):
    IDLE = auto()
    MOVE = auto()
    CLICK_ENGAGED = auto()
    DRAG = auto()
    SCROLL = auto()
    COOLDOWN = auto()


@dataclass
class StateInfo:
    current_state: InteractionState
    previous_state: InteractionState
    state_duration: float
    last_action: str
    pending_action: str


class StateManager:
    def __init__(self):
        self.current_state = InteractionState.IDLE
        self.previous_state = InteractionState.IDLE
        self.state_start_time = 0.0
        self.last_action = "none"
        self.pending_action = "none"
        self.cooldown_until = 0.0
        self._click_latched = False
        self._move_latched = False
        self._drag_active = False
        self._move_start_threshold = 0.008
        self._move_stop_threshold = 0.0055
        self._move_stop_threshold_snappy = 0.0062
        self._click_approach_distance = config.PINCH_THRESHOLD * 1.33

    def update(self, temporal_features, timestamp: float) -> StateInfo:
        self.pending_action = "none"

        if not temporal_features.is_pinched:
            self._click_latched = False

        if self.current_state == InteractionState.COOLDOWN:
            if timestamp < self.cooldown_until:
                return self._build_state_info(timestamp)
            self._transition(InteractionState.IDLE, timestamp)

        if self._drag_active and not temporal_features.is_pinched:
            self.pending_action = "drag_end"
            self.last_action = "drag_end"
            self._drag_active = False
            self._transition(InteractionState.IDLE, timestamp)
            return self._build_state_info(timestamp)

        if self._is_drag_gesture(temporal_features):
            self._transition(InteractionState.DRAG, timestamp)
            if not self._drag_active:
                self.pending_action = "drag_start"
                self.last_action = "drag_start"
                self._drag_active = True
            else:
                self.pending_action = "drag_update"
            return self._build_state_info(timestamp)

        if self._is_click_gesture(temporal_features):
            self._transition(InteractionState.CLICK_ENGAGED, timestamp)
            if (
                temporal_features.pinch_duration >= config.CLICK_HOLD_TIME
                and not self._click_latched
            ):
                self.pending_action = "click"
                self.last_action = "click"
                self._click_latched = True
        elif self._is_move_gesture(temporal_features):
            self._transition(InteractionState.MOVE, timestamp)
        else:
            self._transition(InteractionState.IDLE, timestamp)

        return self._build_state_info(timestamp)

    def enter_cooldown(self, timestamp: float):
        self._transition(InteractionState.COOLDOWN, timestamp)
        self.cooldown_until = timestamp + config.COOLDOWN_TIME

    def reset(self):
        self.current_state = InteractionState.IDLE
        self.previous_state = InteractionState.IDLE
        self.state_start_time = 0.0
        self.pending_action = "none"
        self.cooldown_until = 0.0
        self._click_latched = False
        self._move_latched = False
        self._drag_active = False

    def _is_click_gesture(self, temporal_features) -> bool:
        return bool(temporal_features.is_pinched and temporal_features.palm_stability_trend > 0.6)

    def _is_move_gesture(self, temporal_features) -> bool:
        if temporal_features.is_pinched:
            self._move_latched = False
            return False

        velocity = temporal_features.palm_velocity_trend
        thumb_index_distance = float(getattr(temporal_features, "thumb_index_distance", 1.0))

        # Keep slower precision approach for click posture, tighten stop elsewhere.
        if thumb_index_distance <= self._click_approach_distance:
            effective_stop_threshold = self._move_stop_threshold
        else:
            effective_stop_threshold = self._move_stop_threshold_snappy

        if self._move_latched:
            moving = velocity > effective_stop_threshold
        else:
            moving = velocity > self._move_start_threshold

        self._move_latched = moving
        return bool(moving)

    def _is_drag_gesture(self, temporal_features) -> bool:
        min_drag_hold = max(config.CLICK_HOLD_TIME + 0.25, 0.4)
        return bool(
            temporal_features.is_pinched
            and temporal_features.pinch_duration >= min_drag_hold
            and temporal_features.palm_velocity_trend > self._move_stop_threshold
        )

    def _transition(self, new_state: InteractionState, timestamp: float):
        if self.current_state == new_state:
            return
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = timestamp

    def _build_state_info(self, timestamp: float) -> StateInfo:
        duration = max(0.0, timestamp - self.state_start_time)
        return StateInfo(
            current_state=self.current_state,
            previous_state=self.previous_state,
            state_duration=duration,
            last_action=self.last_action,
            pending_action=self.pending_action,
        )
