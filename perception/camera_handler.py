import cv2

import config


class CameraHandler:
    def __init__(self, camera_id: int = config.CAMERA_ID):
        self.camera_id = camera_id
        self.capture = None

    def open(self) -> bool:
        candidate_ids = [self.camera_id] + [idx for idx in range(4) if idx != self.camera_id]

        for candidate in candidate_ids:
            cap = cv2.VideoCapture(candidate)
            if not cap.isOpened():
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

            ok, _ = cap.read()
            if not ok:
                cap.release()
                continue

            self.capture = cap
            self.camera_id = candidate
            return True

        return False

    def get_frame(self):
        if self.capture is None:
            return False, None

        success, frame = self.capture.read()
        if not success:
            return False, None

        frame = cv2.flip(frame, 1)
        return True, frame

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
