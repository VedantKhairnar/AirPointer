"""
Debug Visualization for AirPointer
Overlay state, landmarks, and debug info on video feed
"""

import cv2
import numpy as np


class DebugVisualizer:
    def __init__(self, frame_width=1280, frame_height=720):
        """Initialize debug visualizer"""
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Colors
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_CYAN = (255, 255, 0)
        
        # Font
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.5
        self.FONT_THICKNESS = 1
    
    def draw_state_info(self, frame, state_info, temporal_features):
        """
        Draw state and feature information on frame.
        
        Args:
            frame: Input frame
            state_info: StateInfo object
            temporal_features: TemporalFeatures object
        
        Returns:
            Annotated frame
        """
        # State box (top-left)
        state_color = self._get_state_color(state_info.current_state.value)
        cv2.rectangle(frame, (10, 10), (300, 150), state_color, 2)
        
        y_pos = 30
        cv2.putText(frame, f"State: {state_info.current_state.value}", (20, y_pos),
                   self.FONT, self.FONT_SCALE + 0.2, state_color, 2)
        
        y_pos += 25
        cv2.putText(frame, f"Duration: {state_info.state_duration:.2f}s", (20, y_pos),
                   self.FONT, self.FONT_SCALE, self.COLOR_WHITE, 1)
        
        y_pos += 20
        cv2.putText(frame, f"Action: {state_info.last_action or 'None'}", (20, y_pos),
                   self.FONT, self.FONT_SCALE, self.COLOR_WHITE, 1)
        
        y_pos += 20
        cv2.putText(frame, f"Confidence: {temporal_features.confidence:.2f}", (20, y_pos),
                   self.FONT, self.FONT_SCALE, self.COLOR_WHITE, 1)
        
        # Motion info (top-right)
        y_pos = 30
        cv2.putText(frame, f"Velocity: {temporal_features.palm_velocity_trend:.3f}", 
                   (frame.shape[1] - 350, y_pos),
                   self.FONT, self.FONT_SCALE, self.COLOR_WHITE, 1)
        
        y_pos += 20
        cv2.putText(frame, f"Stability: {temporal_features.palm_stability_trend:.2f}", 
                   (frame.shape[1] - 350, y_pos),
                   self.FONT, self.FONT_SCALE, self.COLOR_WHITE, 1)
        
        y_pos += 20
        pinch_status = "PINCHED" if temporal_features.is_pinched else "open"
        cv2.putText(frame, f"Pinch: {pinch_status} ({temporal_features.current_features.thumb_index_distance:.2f})", 
                   (frame.shape[1] - 350, y_pos),
                   self.FONT, self.FONT_SCALE, 
                   self.COLOR_RED if temporal_features.is_pinched else self.COLOR_GREEN, 1)
        
        y_pos += 20
        moving = "MOVING" if temporal_features.is_moving else "STABLE"
        cv2.putText(frame, f"Motion: {moving}", 
                   (frame.shape[1] - 350, y_pos),
                   self.FONT, self.FONT_SCALE, self.COLOR_YELLOW, 1)
        
        return frame
    
    def draw_palm_indicator(self, frame, hand_features):
        """
        Draw palm position and motion indicator.
        
        Args:
            frame: Input frame
            hand_features: HandFeatures object
        
        Returns:
            Annotated frame
        """
        # Convert normalized coords to pixel coords
        palm_x = int(hand_features.palm_x * self.frame_width)
        palm_y = int(hand_features.palm_y * self.frame_height)
        
        # Draw palm center
        cv2.circle(frame, (palm_x, palm_y), 10, self.COLOR_CYAN, 2)
        
        # Draw velocity vector
        velocity_scale = 50  # Scale velocity for visualization
        velocity_magnitude = min(hand_features.palm_velocity * velocity_scale, 100)
        
        # Draw motion trail (simplified)
        if velocity_magnitude > 0.5:
            endpoint_x = int(palm_x + velocity_magnitude)
            endpoint_y = palm_y
            cv2.arrowedLine(frame, (palm_x, palm_y), (endpoint_x, endpoint_y),
                           self.COLOR_YELLOW, 2, tipLength=0.3)
        
        # Stability indicator (ring around palm)
        stability_color = self.COLOR_GREEN if hand_features.palm_stability > 0.7 else self.COLOR_YELLOW
        cv2.circle(frame, (palm_x, palm_y), 20, stability_color, 1)
        
        return frame
    
    def draw_finger_states(self, frame, hand_features):
        """
        Draw finger extension states.
        
        Args:
            frame: Input frame
            hand_features: HandFeatures object
        
        Returns:
            Annotated frame
        """
        # Bottom-left corner for finger indicators
        x_start = 20
        y_start = frame.shape[0] - 80
        
        fingers = [
            ("Thumb", hand_features.thumb_extended),
            ("Index", hand_features.index_extended),
            ("Middle", hand_features.middle_extended),
            ("Ring", hand_features.ring_extended),
            ("Pinky", hand_features.pinky_extended),
        ]
        
        for i, (name, is_ext) in enumerate(fingers):
            color = self.COLOR_GREEN if is_ext else self.COLOR_RED
            y_pos = y_start + i * 15
            
            circle_x = x_start + 10
            cv2.circle(frame, (circle_x, y_pos), 5, color, -1)
            cv2.putText(frame, f"{name}: {'EXT' if is_ext else 'BENT'}", 
                       (x_start + 30, y_pos + 5),
                       self.FONT, 0.4, self.COLOR_WHITE, 1)
        
        return frame
    
    def _get_state_color(self, state_name):
        """Get color for state"""
        colors = {
            "IDLE": self.COLOR_BLUE,
            "MOVE": self.COLOR_GREEN,
            "CLICK_ENGAGED": self.COLOR_RED,
            "DRAG": self.COLOR_CYAN,
            "SCROLL": self.COLOR_YELLOW,
            "COOLDOWN": (128, 128, 128),
        }
        return colors.get(state_name, self.COLOR_WHITE)
    
    def draw_all_debug_info(self, frame, hand_features, state_info, temporal_features):
        """
        Draw all debug information on frame.
        
        Args:
            frame: Input frame
            hand_features: HandFeatures object
            state_info: StateInfo object
            temporal_features: TemporalFeatures object
        
        Returns:
            Annotated frame with all overlays
        """
        frame = self.draw_state_info(frame, state_info, temporal_features)
        frame = self.draw_palm_indicator(frame, hand_features)
        frame = self.draw_finger_states(frame, hand_features)
        return frame
