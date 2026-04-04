import time

import cv2

import config
from action_execution.executor import ActionExecutor
from debug.visualizer import DebugVisualizer
from feature_extraction.extractor import FeatureExtractor
from perception.camera_handler import CameraHandler
from perception.hand_detector import HandDetector
from state_management.state_machine import InteractionState, StateManager
from temporal_analysis.analyzer import TemporalAnalyzer


def main():
    camera = CameraHandler()
    detector = HandDetector(model_path=config.MEDIAPIPE_MODEL_PATH)
    extractor = FeatureExtractor()
    analyzer = TemporalAnalyzer(window_size=config.TEMPORAL_WINDOW_SIZE)
    state_manager = StateManager()
    executor = ActionExecutor()
    visualizer = DebugVisualizer()

    debug_mode = config.DEBUG_MODE_DEFAULT
    mouse_enabled = config.MOUSE_ENABLED_DEFAULT

    if not camera.open():
        raise RuntimeError("Unable to open camera")

    try:
        while True:
            ok, frame = camera.get_frame()
            if not ok:
                continue

            ts = time.time()
            landmarks, _ = detector.detect(frame)

            if landmarks is None:
                if executor.is_currently_dragging():
                    executor.end_drag()
                executor.reset_hand_tracking()
                extractor.reset()
                analyzer.reset()
                state_manager.reset()
                if debug_mode:
                    cv2.putText(
                        frame,
                        "No hand detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("AirPointer", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    debug_mode = not debug_mode
                    if not debug_mode:
                        cv2.destroyWindow("AirPointer")
                if key == ord("m"):
                    mouse_enabled = not mouse_enabled
                continue

            if debug_mode and config.SHOW_LANDMARKS:
                frame = detector.draw_landmarks(frame, landmarks)

            hand_features = extractor.extract_features(landmarks, ts)
            temporal = analyzer.update(hand_features)
            state = state_manager.update(temporal, ts)

            if mouse_enabled:
                if state.current_state == InteractionState.DRAG:
                    if state.pending_action == "drag_start":
                        executor.start_drag(hand_features)
                    else:
                        executor.update_drag(hand_features)
                elif executor.is_currently_dragging():
                    executor.end_drag()

                if state.current_state == InteractionState.MOVE:
                    executor.execute_move(hand_features)
                if (
                    state.current_state == InteractionState.CLICK_ENGAGED
                    and state.pending_action == "click"
                ):
                    executor.execute_click()
                    state_manager.enter_cooldown(ts)

            if debug_mode:
                frame = visualizer.draw_all_debug_info(frame, hand_features, temporal, state)
                cv2.imshow("AirPointer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug_mode = not debug_mode
                if not debug_mode:
                    cv2.destroyWindow("AirPointer")
            if key == ord("m"):
                mouse_enabled = not mouse_enabled
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
