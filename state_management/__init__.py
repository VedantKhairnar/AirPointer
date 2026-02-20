"""
State Management module for AirPointer
- Finite state machine for interaction intent
- Handles state transitions and cooldowns
"""

from .state_machine import StateManager, InteractionState, StateInfo

__all__ = ['StateManager', 'InteractionState', 'StateInfo']
