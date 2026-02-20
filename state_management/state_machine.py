"""
State Management for AirPointer
Finite State Machine for interaction intent
"""

import time
from enum import Enum
from dataclasses import dataclass


class InteractionState(Enum):
    """Interaction intent states"""
    IDLE = "IDLE"
    MOVE = "MOVE"
    CLICK_ENGAGED = "CLICK_ENGAGED"
    DRAG = "DRAG"
    SCROLL = "SCROLL"
    COOLDOWN = "COOLDOWN"


@dataclass
class StateInfo:
    """Current state information"""
    current_state: InteractionState
    previous_state: InteractionState
    state_duration: float  # Seconds in current state
    last_action: str  # Last action triggered
    pending_action: str  # Action waiting to be executed


class StateManager:
    def __init__(self):
        """Initialize state manager with FSM"""
        self.current_state = InteractionState.IDLE
        self.previous_state = InteractionState.IDLE
        self.state_start_time = time.time()
        self.last_action = None
        self.pending_action = None
        
        # Temporal thresholds
        self.CLICK_HOLD_TIME = 0.15  # 150 ms to confirm click
        self.COOLDOWN_TIME = 0.2  # 200 ms cooldown after action
        self.DRAG_MIN_TIME = 0.1  # 100 ms minimum drag duration
        self.SCROLL_VELOCITY_THRESHOLD = 0.02
        
        # Action tracking
        self.cooldown_until = 0
        self.click_start_time = None
        self.drag_start_time = None
    
    def update(self, temporal_features):
        """
        Update state based on temporal features.
        
        Args:
            temporal_features: TemporalFeatures object
        
        Returns:
            StateInfo object
        """
        current_time = time.time()
        state_duration = current_time - self.state_start_time
        
        # Handle cooldown expiration
        if self.current_state == InteractionState.COOLDOWN:
            if current_time >= self.cooldown_until:
                self.current_state = InteractionState.IDLE
                self.state_start_time = current_time
        
        # In cooldown, don't process
        if self.current_state == InteractionState.COOLDOWN:
            return StateInfo(
                current_state=self.current_state,
                previous_state=self.previous_state,
                state_duration=state_duration,
                last_action=self.last_action,
                pending_action=None
            )
        
        # FSM Logic with priority arbitration
        self.pending_action = None
        
        # Priority 1: DRAG (sustained finger engagement + palm movement)
        if self._should_drag(temporal_features):
            self._transition_to(InteractionState.DRAG, current_time)
        
        # Priority 2: CLICK (stable palm + deliberate finger action)
        elif self._should_click(temporal_features, current_time):
            self._transition_to(InteractionState.CLICK_ENGAGED, current_time)
            self.pending_action = "click"
        
        # Priority 3: SCROLL (vertical motion)
        elif self._should_scroll(temporal_features):
            self._transition_to(InteractionState.SCROLL, current_time)
        
        # Priority 4: MOVE (smooth palm movement)
        elif self._should_move(temporal_features):
            self._transition_to(InteractionState.MOVE, current_time)
        
        # Default: IDLE
        else:
            self._transition_to(InteractionState.IDLE, current_time)
        
        return StateInfo(
            current_state=self.current_state,
            previous_state=self.previous_state,
            state_duration=state_duration,
            last_action=self.last_action,
            pending_action=self.pending_action
        )
    
    def _should_move(self, temporal_features):
        """Check if should be in MOVE state"""
        # Palm moving, no finger engagement
        is_moving = temporal_features.palm_velocity_trend > 0.008
        no_pinch = not temporal_features.is_pinched
        low_finger_motion = temporal_features.finger_motion_magnitude < 0.01
        
        return is_moving and no_pinch and low_finger_motion
    
    def _should_click(self, temporal_features, current_time):
        """Check if should trigger CLICK"""
        # Palm stable, index finger extends/retracts deliberately
        palm_stable = temporal_features.palm_stability_trend > 0.75
        
        # Check if index finger motion is deliberate (not noise)
        finger_motion = temporal_features.finger_motion_magnitude
        has_deliberate_motion = 0.015 < finger_motion < 0.05
        
        # Click must be held for minimum time
        if palm_stable and has_deliberate_motion:
            if self.click_start_time is None:
                self.click_start_time = current_time
            
            hold_time = current_time - self.click_start_time
            if hold_time >= self.CLICK_HOLD_TIME:
                return True
        else:
            self.click_start_time = None
        
        return False
    
    def _should_drag(self, temporal_features):
        """Check if should be in DRAG state"""
        # Pinch sustained + palm moving
        pinch_sustained = temporal_features.is_pinched and temporal_features.pinch_duration > 0.1
        palm_moving = temporal_features.palm_velocity_trend > 0.008
        
        return pinch_sustained and palm_moving
    
    def _should_scroll(self, temporal_features):
        """Check if should be in SCROLL state"""
        # Vertical motion detected (hand moving up/down significantly)
        # This requires tracking Y-axis motion specifically
        palm_moving = temporal_features.palm_velocity_trend > 0.01
        no_pinch = not temporal_features.is_pinched
        
        # Simplified: detect scroll if moving and not pinched
        # (In real implementation, check Y-axis specifically)
        return palm_moving and no_pinch
    
    def _transition_to(self, new_state, current_time):
        """Handle state transition"""
        if new_state != self.current_state:
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_start_time = current_time
    
    def enter_cooldown(self, current_time):
        """Enter cooldown state after action"""
        self.current_state = InteractionState.COOLDOWN
        self.cooldown_until = current_time + self.COOLDOWN_TIME
        self.state_start_time = current_time
