"""
Action Execution for AirPointer
Maps states to OS-level mouse/trackpad events
"""

import pyautogui
import time


class ActionExecutor:
    def __init__(self, screen_width=1920, screen_height=1080, frame_width=1280, frame_height=720):
        """
        Initialize action executor.
        
        Args:
            screen_width: Screen resolution width
            screen_height: Screen resolution height
            frame_width: Camera frame width
            frame_height: Camera frame height
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Scaling factors (hand position to cursor position)
        self.scale_x = screen_width / frame_width
        self.scale_y = screen_height / frame_height
        
        # Smoothing
        self.prev_cursor_x = 0
        self.prev_cursor_y = 0
        self.smoothing_factor = 0.6
        
        # Action tracking
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Disable PyAutoGUI safety pauses for better performance
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.01
    
    def execute_move(self, hand_features):
        """
        Execute cursor movement based on palm position.
        
        Args:
            hand_features: HandFeatures object
        """
        # Map normalized hand coordinates (0-1) to screen coordinates
        raw_x = hand_features.palm_x * self.screen_width
        raw_y = hand_features.palm_y * self.screen_height
        
        # Apply smoothing to reduce jitter
        smooth_x = self.prev_cursor_x * (1 - self.smoothing_factor) + raw_x * self.smoothing_factor
        smooth_y = self.prev_cursor_y * (1 - self.smoothing_factor) + raw_y * self.smoothing_factor
        
        # Clamp to screen bounds
        cursor_x = max(0, min(self.screen_width - 1, smooth_x))
        cursor_y = max(0, min(self.screen_height - 1, smooth_y))
        
        # Move cursor
        pyautogui.moveTo(cursor_x, cursor_y)
        
        # Update previous position
        self.prev_cursor_x = cursor_x
        self.prev_cursor_y = cursor_y
    
    def execute_click(self, button='left'):
        """
        Execute mouse click.
        
        Args:
            button: 'left' or 'right'
        """
        pyautogui.click(button=button)
    
    def execute_double_click(self, button='left'):
        """Execute double click"""
        pyautogui.doubleClick(button=button)
    
    def start_drag(self, hand_features):
        """
        Start drag operation.
        
        Args:
            hand_features: HandFeatures object
        """
        if not self.is_dragging:
            self.is_dragging = True
            self.drag_start_x = hand_features.palm_x * self.screen_width
            self.drag_start_y = hand_features.palm_y * self.screen_height
            
            # Begin mouse press
            pyautogui.mouseDown(button='left')
    
    def update_drag(self, hand_features):
        """
        Update drag operation with new position.
        
        Args:
            hand_features: HandFeatures object
        """
        if self.is_dragging:
            # Update cursor position (same as move, but while pressed)
            self.execute_move(hand_features)
    
    def end_drag(self):
        """End drag operation"""
        if self.is_dragging:
            self.is_dragging = False
            pyautogui.mouseUp(button='left')
    
    def execute_scroll(self, direction='up', amount=3):
        """
        Execute scroll action.
        
        Args:
            direction: 'up' or 'down'
            amount: Number of scroll clicks
        """
        scroll_value = amount if direction == 'down' else -amount
        pyautogui.scroll(scroll_value)
    
    def is_currently_dragging(self):
        """Check if currently in drag state"""
        return self.is_dragging
