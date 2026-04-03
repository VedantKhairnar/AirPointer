"""
AirPointer: Advanced Configuration-Driven Version
Uses config.py for easy tuning without modifying main code
"""

import cv2
import time
import sys
import argparse
import config
import logging
from types import SimpleNamespace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("application.log"),
        logging.StreamHandler()
    ]
)

logging.info("Application started.")

from perception.camera_handler import CameraHandler
from perception.hand_detector import HandDetector
from feature_extraction.extractor import FeatureExtractor
from temporal_analysis.analyzer import TemporalAnalyzer
from state_management.state_machine import StateManager, InteractionState
from action_execution.executor import ActionExecutor
from debug.visualizer import DebugVisualizer
from model_loader import initialize_models
from inference_pipeline import process_frame


class AirPointerSystem:
    def __init__(self, default_model="stage_one"):
        """Initialize all components of AirPointer"""
        print("[AirPointer] Initializing system...")
        logging.info(f"Initializing AirPointerSystem with default_model={default_model}")

        # Store model selection
        self.default_model = default_model

        # Initialize perception
        self.camera = CameraHandler(
            camera_id=config.CAMERA_ID,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=config.CAMERA_FPS
        )
        # Only initialize HandDetector if using mediapipe
        self.detector = None
        if self.default_model == "mediapipe":
            self.detector = HandDetector(
                max_num_hands=config.MEDIAPIPE_MAX_HANDS
            )
            logging.info("HandDetector (MediaPipe) initialized.")

        # Initialize processing pipeline
        self.feature_extractor = FeatureExtractor()
        self.temporal_analyzer = TemporalAnalyzer(
            window_size=config.TEMPORAL_WINDOW_SIZE,
            fps=config.TEMPORAL_FPS
        )
        self.state_manager = StateManager()

        # Initialize action execution and visualization
        self.action_executor = ActionExecutor(
            screen_width=config.SCREEN_WIDTH,
            screen_height=config.SCREEN_HEIGHT,
            frame_width=config.CAMERA_WIDTH,
            frame_height=config.CAMERA_HEIGHT,
            sensitivity=config.CURSOR_SENSITIVITY
        )
        self.visualizer = DebugVisualizer(
            frame_width=config.CAMERA_WIDTH,
            frame_height=config.CAMERA_HEIGHT
        )

        # State variables
        self.debug_mode = config.DEBUG_MODE_DEFAULT
        self.mouse_enabled = config.MOUSE_ENABLED_DEFAULT
        self.frame_count = 0
        self.start_time = None
    
    def run(self):
        """Main loop"""
        try:
            self.camera.open()
            self.start_time = time.time()
            
            print("[AirPointer] System ready. Camera opened.")
            print("[Controls]")
            print("  q - Quit")
            print("  d - Toggle debug mode")
            print("  m - Toggle mouse control")
            print()
            

            # Initialize models
            models, device = initialize_models("config/models.json")
            if self.default_model not in models:
                logging.error(f"Default model '{self.default_model}' not found. Using 'stage_one'.")
                print(f"[Error] Default model '{self.default_model}' not found. Using 'stage_one'.")
                self.default_model = "stage_one"
            print(f"Switched to model: {self.default_model}")
            logging.info(f"Switched to model: {self.default_model}")

            print("Webcam feed started. Press 'q' to quit.")

            while True:
                # Capture frame
                success, frame = self.camera.get_frame()
                if not success:
                    logging.error("Failed to capture frame")
                    print("[Error] Failed to capture frame")
                    break

                self.frame_count += 1
                current_timestamp = time.time() - self.start_time

                if self.default_model == "mediapipe":
                    # Use MediaPipe for detection
                    landmarks, confidence = self.detector.detect(frame)
                    logging.info(f"[Mediapipe] Detection result: landmarks={'found' if landmarks is not None else 'none'}")
                    if landmarks is None:
                        self.action_executor.reset_hand_tracking()
                        if self.debug_mode:
                            cv2.putText(frame, "No hand detected", (50, 100),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            self._display_frame(frame)
                        if not self._handle_input():
                            break
                        continue
                    # Draw landmarks
                    if self.debug_mode and config.SHOW_LANDMARKS:
                        frame = self.detector.draw_landmarks(frame, landmarks)
                    # Feature extraction and pipeline
                    hand_features = self.feature_extractor.extract_features(landmarks, current_timestamp)
                    temporal_features = self.temporal_analyzer.update(hand_features)
                    state_info = self.state_manager.update(temporal_features)
                    self._execute_actions(state_info, hand_features, temporal_features)
                    if self.debug_mode:
                        frame = self._add_debug_overlays(frame, hand_features, state_info, temporal_features)
                    self._display_frame(frame)
                elif self.default_model == "custom1":
                    # Use custom1 pipeline (stage_one + stage_two)
                    try:
                        logging.info(f"[Custom1 Pipeline] About to call process_frame. Model keys: {list(models['custom1'].keys())}, Device: {device}")
                        frame, inference_meta = process_frame(
                            frame,
                            models["custom1"],
                            device,
                            hand_features=None,
                            return_metadata=True,
                            detection_threshold=getattr(config, "CUSTOM_STAGE_ONE_SCORE_THRESHOLD", 0.35)
                        )
                        if inference_meta["detected"] and inference_meta["cursor_point"] is not None:
                            cursor_x, cursor_y = inference_meta["cursor_point"]
                            hand_features = SimpleNamespace(
                                palm_x=float(cursor_x) / float(config.CAMERA_WIDTH),
                                palm_y=float(cursor_y) / float(config.CAMERA_HEIGHT)
                            )
                            if self.mouse_enabled:
                                self.action_executor.execute_move(hand_features)
                        else:
                            self.action_executor.reset_hand_tracking()
                            logging.info(f"[Custom1 Pipeline] No hand detected. score={inference_meta['score']}")
                        logging.info("[Custom1 Pipeline] process_frame executed.")
                    except Exception as e:
                        logging.error(f"[Custom1 Pipeline] process_frame error: {e}")
                        print(f"[Error] process_frame: {e}")
                    self._display_frame(frame)

                # Input handling
                if not self._handle_input():
                    break
        
        except KeyboardInterrupt:
            print("\n[AirPointer] Interrupted by user")
        except Exception as e:
            print(f"[Error] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()
    
    def _execute_actions(self, state_info, hand_features, temporal_features):
        """Execute actions based on current state"""
        if state_info.current_state == InteractionState.MOVE:
            if self.mouse_enabled:
                self.action_executor.execute_move(hand_features)
        
        elif state_info.current_state == InteractionState.CLICK_ENGAGED:
            if state_info.pending_action == "click" and self.mouse_enabled:
                self.action_executor.execute_click(button='left')
                self.state_manager.last_action = "LEFT_CLICK"
                self.state_manager.enter_cooldown(time.time())
    
    def _add_debug_overlays(self, frame, hand_features, state_info, temporal_features):
        """Add all debug information to frame"""
        frame = self.visualizer.draw_all_debug_info(
            frame, hand_features, state_info, temporal_features
        )
        
        # Mouse status
        mouse_status = "MOUSE: ON" if self.mouse_enabled else "MOUSE: OFF"
        status_color = (0, 255, 0) if self.mouse_enabled else (0, 0, 255)
        cv2.putText(frame, mouse_status, (frame.shape[1] - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # FPS counter
        if config.ENABLE_FPS_COUNTER and self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def _display_frame(self, frame):
        """Display frame if debug mode is on"""
        if self.debug_mode:
            cv2.imshow("AirPointer", frame)
    
    def _handle_input(self):
        """Handle keyboard input. Returns False if should quit"""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('d'):
            self.debug_mode = not self.debug_mode
            print(f"[Debug] Mode: {'ON' if self.debug_mode else 'OFF'}")
        elif key == ord('m'):
            self.mouse_enabled = not self.mouse_enabled
            status = "ENABLED" if self.mouse_enabled else "DISABLED"
            print(f"[Mouse Control] {status}")
        return True
    
    def shutdown(self):
        """Clean shutdown"""
        print("[AirPointer] Shutting down...")
        self.camera.close()
        cv2.destroyAllWindows()
        print("[AirPointer] Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="AirPointer Advanced")
    parser.add_argument('--model', type=str, default="custom1", choices=["custom1", "mediapipe"], help="Default model to use: custom1 or mediapipe")
    args = parser.parse_args()
    system = AirPointerSystem(default_model=args.model)
    system.run()


if __name__ == "__main__":
    main()
