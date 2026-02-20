"""
Camera Handler for AirPointer
Manages camera input and frame capture
"""

import cv2


class CameraHandler:
    def __init__(self, camera_id=0, width=1280, height=720, fps=30):
        """
        Initialize camera handler.
        
        Args:
            camera_id: Camera device ID (default: 0)
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_open = False
    
    def open(self):
        """Open camera and configure settings."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.is_open = True
        print(f"Camera opened: {self.width}x{self.height} @ {self.fps}fps")
    
    def get_frame(self):
        """
        Capture a frame from camera.
        
        Returns:
            (success: bool, frame: numpy.ndarray)
        """
        if not self.is_open:
            return False, None
        
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)  # Horizontal flip for mirror effect
        return success, frame
    
    def close(self):
        """Close camera."""
        if self.cap:
            self.cap.release()
            self.is_open = False
            print("Camera closed")
