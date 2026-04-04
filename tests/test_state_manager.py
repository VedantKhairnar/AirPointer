from state_management.state_machine import InteractionState, StateManager
from temporal_analysis.analyzer import TemporalFeatures


def _temporal(pinched: bool, pinch_duration: float) -> TemporalFeatures:
    return TemporalFeatures(
        palm_velocity_trend=0.02,
        palm_acceleration_trend=0.0,
        palm_stability_trend=0.9,
        finger_motion_magnitude=0.0,
        finger_consistency=1.0,
        is_moving=True,
        is_stable=True,
        is_transitioning=False,
        is_pinched=pinched,
        pinch_duration=pinch_duration,
        confidence=0.9,
    )


def test_click_latched_until_release():
    manager = StateManager()

    state1 = manager.update(_temporal(pinched=True, pinch_duration=0.16), 1.0)
    assert state1.current_state == InteractionState.CLICK_ENGAGED
    assert state1.pending_action == "click"

    state2 = manager.update(_temporal(pinched=True, pinch_duration=0.30), 1.1)
    assert state2.current_state == InteractionState.CLICK_ENGAGED
    assert state2.pending_action == "none"

    manager.update(_temporal(pinched=False, pinch_duration=0.0), 1.2)
    state3 = manager.update(_temporal(pinched=True, pinch_duration=0.20), 1.3)
    assert state3.pending_action == "click"


def test_move_hysteresis_prevents_premature_idle():
    manager = StateManager()

    moving = _temporal(pinched=False, pinch_duration=0.0)
    moving.palm_velocity_trend = 0.010
    state1 = manager.update(moving, 1.0)
    assert state1.current_state == InteractionState.MOVE

    slow = _temporal(pinched=False, pinch_duration=0.0)
    slow.palm_velocity_trend = 0.006
    state2 = manager.update(slow, 1.1)
    assert state2.current_state == InteractionState.MOVE

    very_slow = _temporal(pinched=False, pinch_duration=0.0)
    very_slow.palm_velocity_trend = 0.001
    state3 = manager.update(very_slow, 1.2)
    assert state3.current_state == InteractionState.IDLE


def test_drag_start_update_end_flow():
    manager = StateManager()

    drag_start = _temporal(pinched=True, pinch_duration=0.42)
    drag_start.palm_velocity_trend = 0.02
    state1 = manager.update(drag_start, 1.0)
    assert state1.current_state == InteractionState.DRAG
    assert state1.pending_action == "drag_start"

    drag_update = _temporal(pinched=True, pinch_duration=0.55)
    drag_update.palm_velocity_trend = 0.015
    state2 = manager.update(drag_update, 1.1)
    assert state2.current_state == InteractionState.DRAG
    assert state2.pending_action == "drag_update"

    release = _temporal(pinched=False, pinch_duration=0.0)
    state3 = manager.update(release, 1.2)
    assert state3.current_state == InteractionState.IDLE
    assert state3.pending_action == "drag_end"
