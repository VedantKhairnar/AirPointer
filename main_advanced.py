import argparse
import json
import logging
import os
import traceback
from pathlib import Path
import time

import cv2
import numpy as np

import config
from action_execution.executor import ActionExecutor
from debug.visualizer import DebugVisualizer
from feature_extraction.extractor import FeatureExtractor
from inference_pipeline import process_frame
from model_loader import initialize_models
from perception.camera_handler import CameraHandler
from perception.hand_detector import HandDetector
from control_server import ControlStateServer
from preview_server import PreviewStreamServer
from state_management.state_machine import InteractionState, StateManager
from temporal_analysis.analyzer import TemporalAnalyzer


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("airpointer")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler("application.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class AirPointerSystem:
    def __init__(self, selected_model: str = "custom1", embedded_ui: bool = False, preview_port: int = 8765):
        self.logger = setup_logger()
        self.selected_model = selected_model
        self.embedded_ui = embedded_ui
        self.preview_port = preview_port

        self.camera = CameraHandler()
        self.detector = None
        self.extractor = FeatureExtractor()
        self.analyzer = TemporalAnalyzer(window_size=config.TEMPORAL_WINDOW_SIZE)
        self.state_manager = StateManager()
        self.executor = ActionExecutor()
        self.visualizer = DebugVisualizer()

        self.debug_mode = config.DEBUG_MODE_DEFAULT
        self.mouse_enabled = config.MOUSE_ENABLED_DEFAULT

        self.models = None
        self.device = None
        self._last_detection_log_time = 0.0
        self._hand_visible = False
        self._warmup_frames_remaining = 0
        self._landmark_ema = None
        self._landmark_alpha = 0.35
        self._missed_frames = 0
        self._max_missed_frames = 4
        self.preview_server = None
        self.control_server = None
        self.control_port = int(os.environ.get("AIRPOINTER_CONTROL_PORT", "8766"))

    def initialize(self):
        self.logger.info("Starting AirPointer system")
        self.logger.info("Selected model: %s", self.selected_model)

        if not self.camera.open():
            raise RuntimeError("Unable to open camera")

        config_path = Path("config/models.json")
        with config_path.open("r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        self.logger.info("Model config keys: %s", list(raw_cfg.keys()))

        self.models, self.device = initialize_models(str(config_path))

        if self.selected_model not in self.models:
            self.logger.warning(
                "Invalid model '%s'. Falling back to 'custom1'.", self.selected_model
            )
            self.selected_model = "custom1"

        if "custom1" in raw_cfg:
            s1 = raw_cfg["custom1"]["stage_one"]
            s2 = raw_cfg["custom1"]["stage_two"]
            self.logger.info(
                "stage_one path=%s loader=%s", s1["path"], s1["model_fn"]
            )
            self.logger.info(
                "stage_two path=%s loader=%s", s2["path"], s2["model_fn"]
            )

        if "custom1" in self.models:
            self.logger.info(
                "stage_one weights loaded=%s (%s)",
                self.models["custom1"]["stage_one_loaded"],
                self.models["custom1"]["stage_one_path"],
            )
            self.logger.info(
                "stage_two weights loaded=%s (%s)",
                self.models["custom1"]["stage_two_loaded"],
                self.models["custom1"]["stage_two_path"],
            )

        if self.selected_model == "mediapipe":
            self.detector = HandDetector(model_path=config.MEDIAPIPE_MODEL_PATH)

        if self.embedded_ui:
            self.preview_server = PreviewStreamServer(port=self.preview_port)
            self.preview_server.start()
            self.control_server = ControlStateServer(port=self.control_port)
            self.control_server.start()
            self.control_server.update_state({
                "debug_mode": self.debug_mode,
                "mouse_enabled": self.mouse_enabled,
                "running": True,
            })
            self.logger.info("Embedded preview stream started at %s", self.preview_server.url)
            self.logger.info("Embedded control API started at %s", self.control_server.url)

    def run(self):
        try:
            while True:
                ok, frame = self.camera.get_frame()
                if not ok:
                    continue

                ts = time.time()

                if self.embedded_ui and self.control_server is not None:
                    control_state = self.control_server.get_state()
                    self.debug_mode = control_state.get("debug_mode", self.debug_mode)
                    self.mouse_enabled = control_state.get("mouse_enabled", self.mouse_enabled)

                if self.selected_model == "mediapipe":
                    rendered = self._process_mediapipe(frame, ts)
                else:
                    rendered = self._process_custom1(frame)

                if self.preview_server is not None and rendered is not None:
                    self.preview_server.update_frame(rendered)

                if not self.embedded_ui:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("d"):
                        self.debug_mode = not self.debug_mode
                        if not self.debug_mode:
                            cv2.destroyWindow("AirPointer")
                    if key == ord("m"):
                        self.mouse_enabled = not self.mouse_enabled

        except Exception as exc:
            self.logger.error("Uncaught exception: %s", exc)
            self.logger.error(traceback.format_exc())
            raise
        finally:
            if self.control_server is not None:
                self.control_server.update_state({"running": False})
                self.control_server.stop()
            if self.preview_server is not None:
                self.preview_server.stop()
            self.camera.close()
            cv2.destroyAllWindows()

    def _process_mediapipe(self, frame, timestamp: float):
        landmarks, _ = self.detector.detect(frame)

        if landmarks is None:
            self._throttled_detection_log("mediapipe detection: none")
            self._missed_frames += 1
            if self._missed_frames > self._max_missed_frames:
                self._reset_tracking_pipeline()
            if self.debug_mode:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
                if not self.embedded_ui:
                    cv2.imshow("AirPointer", frame)
            return frame

        self._missed_frames = 0
        if not self._hand_visible:
            self._hand_visible = True
            self._warmup_frames_remaining = 3

        self._throttled_detection_log("mediapipe detection: found")

        if self.debug_mode and config.SHOW_LANDMARKS:
            frame = self.detector.draw_landmarks(frame, landmarks)

        self._run_shared_pipeline(frame, self._smooth_landmarks(landmarks), timestamp)
        if self.debug_mode:
            if not self.embedded_ui:
                cv2.imshow("AirPointer", frame)
        return frame

    def _process_custom1(self, frame):
        timestamp = time.time()

        frame_out, metadata = process_frame(
            frame,
            self.models["custom1"],
            self.device,
            return_metadata=True,
            detection_threshold=config.CUSTOM_STAGE_ONE_SCORE_THRESHOLD,
        )

        detected = metadata.get("detected", False)
        keypoints = metadata.get("keypoints") or []
        score = metadata.get("score")

        if detected and len(keypoints) >= 21:
            self._missed_frames = 0
            if not self._hand_visible:
                self._hand_visible = True
                self._warmup_frames_remaining = 3

            frame_h, frame_w = frame_out.shape[:2]
            landmarks = np.zeros((21, 3), dtype=np.float32)
            for idx in range(21):
                kp_x, kp_y = keypoints[idx]
                landmarks[idx, 0] = float(np.clip(kp_x / max(1, frame_w), 0.0, 1.0))
                landmarks[idx, 1] = float(np.clip(kp_y / max(1, frame_h), 0.0, 1.0))
                landmarks[idx, 2] = 0.0

            self._run_shared_pipeline(frame_out, self._smooth_landmarks(landmarks), timestamp)
        else:
            self.logger.info("custom1 no-detection score=%s", score)
            self._missed_frames += 1
            if self._missed_frames > self._max_missed_frames:
                self._reset_tracking_pipeline()

            if self.debug_mode:
                cv2.putText(
                    frame_out,
                    "No hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

        if self.debug_mode:
            if not self.embedded_ui:
                cv2.imshow("AirPointer", frame_out)
        return frame_out

    def _run_shared_pipeline(self, frame, landmarks, timestamp: float):
        hand_features = self.extractor.extract_features(landmarks, timestamp)
        temporal = self.analyzer.update(hand_features)
        state = self.state_manager.update(temporal, timestamp)

        can_act = self._warmup_frames_remaining == 0 and temporal.confidence >= 0.25

        if self.mouse_enabled:
            if can_act and state.current_state == InteractionState.DRAG:
                if state.pending_action == "drag_start":
                    self.executor.start_drag(hand_features)
                else:
                    self.executor.update_drag(hand_features)
            elif self.executor.is_currently_dragging():
                self.executor.end_drag()

            if can_act and state.current_state == InteractionState.MOVE:
                self.executor.execute_move(hand_features)
            if (
                can_act
                and
                state.current_state == InteractionState.CLICK_ENGAGED
                and state.pending_action == "click"
                and temporal.confidence >= 0.35
            ):
                self.executor.execute_click()
                self.state_manager.enter_cooldown(timestamp)

        if self._warmup_frames_remaining > 0:
            self._warmup_frames_remaining -= 1

        if self.debug_mode:
            frame = self.visualizer.draw_all_debug_info(frame, hand_features, temporal, state)

    def _reset_tracking_pipeline(self):
        if self.executor.is_currently_dragging():
            self.executor.end_drag()
        self.executor.reset_hand_tracking()
        self.extractor.reset()
        self.analyzer.reset()
        self.state_manager.reset()
        self._hand_visible = False
        self._warmup_frames_remaining = 0
        self._landmark_ema = None
        self._missed_frames = 0

    def _smooth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        if self._landmark_ema is None:
            self._landmark_ema = landmarks.copy()
            return landmarks

        self._landmark_ema = (
            self._landmark_alpha * landmarks + (1.0 - self._landmark_alpha) * self._landmark_ema
        )
        return self._landmark_ema.copy()

    def _throttled_detection_log(self, message: str, interval_sec: float = 1.0):
        now = time.time()
        if now - self._last_detection_log_time >= interval_sec:
            self.logger.info(message)
            self._last_detection_log_time = now


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["custom1", "mediapipe"],
        default="custom1",
        help="Model mode to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    embedded_ui = os.environ.get("AIRPOINTER_EMBEDDED_UI", "0") == "1"
    preview_port = int(os.environ.get("AIRPOINTER_PREVIEW_PORT", "8765"))
    app = AirPointerSystem(selected_model=args.model, embedded_ui=embedded_ui, preview_port=preview_port)
    app.initialize()
    app.run()

if __name__ == "__main__":
    main()
