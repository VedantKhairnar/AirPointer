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
        
        # Priority 1: CLICK (quick pinch gesture with stable palm)
        if self._should_click(temporal_features, current_time):
            self._transition_to(InteractionState.CLICK_ENGAGED, current_time)
            self.pending_action = "click"
        
        # Priority 2: MOVE (smooth palm movement)
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
        # Palm moving, no pinch
        is_moving = temporal_features.palm_velocity_trend > 0.008
        no_pinch = not temporal_features.is_pinched
        
        return is_moving and no_pinch
    
    def _should_click(self, temporal_features, current_time):
        """Check if should trigger CLICK"""
        # Quick pinch gesture triggers click (pinch without sustained movement)
        is_pinched = temporal_features.is_pinched
        palm_stable = temporal_features.palm_velocity_trend < 0.015
        
        # Click triggered by quick pinch while palm is relatively stable
        if is_pinched and palm_stable:
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
        # Drag functionality disabled
        return False
    
    def _should_scroll(self, temporal_features):
        """Check if should be in SCROLL state"""
        # Scroll functionality disabled
        return False
    
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
