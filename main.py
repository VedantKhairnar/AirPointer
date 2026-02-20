"""
AirPointer: Touchless Mouse/Trackpad Replacement
Entry point for the AirPointer system.

Pipeline:
Camera → Perception (MediaPipe) → Feature Extraction → Temporal Analysis 
→ State Management (FSM) → Action Execution → Debug Visualization

Run this script to start the AirPointer system.
"""

import cv2
import time
import sys

from perception.camera_handler import CameraHandler
from perception.hand_detector import HandDetector
from feature_extraction.extractor import FeatureExtractor
from temporal_analysis.analyzer import TemporalAnalyzer
from state_management.state_machine import StateManager, InteractionState
from action_execution.executor import ActionExecutor
from debug.visualizer import DebugVisualizer


def main():
    print("Initializing AirPointer system...")
    
    # Initialize components
    camera = CameraHandler(camera_id=0, width=1280, height=720, fps=30)
    detector = HandDetector(max_num_hands=1)
    feature_extractor = FeatureExtractor()
    temporal_analyzer = TemporalAnalyzer(window_size=15, fps=30)
    state_manager = StateManager()
    action_executor = ActionExecutor(screen_width=1920, screen_height=1080, 
                                    frame_width=1280, frame_height=720)
    visualizer = DebugVisualizer(frame_width=1280, frame_height=720)
    
    try:
        camera.open()
        print("AirPointer running. Press 'q' to quit, 'd' to toggle debug mode, 'm' to toggle mouse control.\n")
        
        debug_mode = True
        mouse_enabled = False
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Capture frame
            success, frame = camera.get_frame()
            if not success:
                print("Failed to capture frame. Exiting.")
                break
            
            frame_count += 1
            current_timestamp = time.time() - start_time
            
            # Step 1: Perception - Detect hand landmarks
            landmarks, confidence = detector.detect(frame)
            
            if landmarks is None:
                # No hand detected
                if debug_mode:
                    cv2.putText(frame, "No hand detected", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if debug_mode:
                    cv2.imshow("AirPointer Debug", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    debug_mode = not debug_mode
                elif key == ord('m'):
                    mouse_enabled = not mouse_enabled
                    status = "ENABLED" if mouse_enabled else "DISABLED"
                    print(f"Mouse control {status}")
                
                continue
            
            # Draw landmarks on debug frame
            if debug_mode:
                frame = detector.draw_landmarks(frame, landmarks)
            
            # Step 2: Feature Extraction - Convert landmarks to features
            hand_features = feature_extractor.extract_features(landmarks, current_timestamp)
            
            # Step 3: Temporal Analysis - Analyze motion patterns
            temporal_features = temporal_analyzer.update(hand_features)
            
            # Step 4: State Management - Determine interaction intent
            state_info = state_manager.update(temporal_features)
            
            # Step 5: Action Execution - Execute appropriate mouse action
            if state_info.current_state == InteractionState.MOVE:
                if mouse_enabled:
                    action_executor.execute_move(hand_features)
            
            elif state_info.current_state == InteractionState.CLICK_ENGAGED:
                if state_info.pending_action == "click" and mouse_enabled:
                    action_executor.execute_click(button='left')
                    state_manager.last_action = "LEFT_CLICK"
                    state_manager.enter_cooldown(time.time())
            
            # Step 6: Debug Visualization
            if debug_mode:
                frame = visualizer.draw_all_debug_info(frame, hand_features, state_info, temporal_features)
                
                # Add mouse status
                mouse_status = "MOUSE: ON" if mouse_enabled else "MOUSE: OFF"
                status_color = (0, 255, 0) if mouse_enabled else (0, 0, 255)
                cv2.putText(frame, mouse_status, (1280 - 250, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # FPS counter
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow("AirPointer Debug", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting AirPointer...")
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            elif key == ord('m'):
                mouse_enabled = not mouse_enabled
                status = "ENABLED" if mouse_enabled else "DISABLED"
                print(f"Mouse control {status}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("AirPointer shutdown complete")


if __name__ == "__main__":
    main()
