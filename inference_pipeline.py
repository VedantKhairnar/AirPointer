import cv2
import time
import torch
import numpy as np

class CursorController:
    def __init__(self):
        self.previous_position = None
        self.smoothing_factor = 0.8  # Adjust for more or less smoothing

    def smooth_cursor(self, current_position):
        if self.previous_position is None:
            self.previous_position = current_position
            return current_position

        smoothed_position = (
            self.smoothing_factor * np.array(self.previous_position)
            + (1 - self.smoothing_factor) * np.array(current_position)
        )
        self.previous_position = smoothed_position.tolist()
        return smoothed_position.tolist()

class GestureController:
    def __init__(self):
        self.click_threshold = 0.2  # Adjust as needed

    def is_pinch_click(self, distance):
        # Simplified logic: Check if distance is below the threshold
        return distance < self.click_threshold

class DragController:
    def __init__(self):
        self.is_dragging = False

    def toggle_drag(self, pinch_distance):
        # Start drag if pinch is detected and not already dragging
        if pinch_distance < 0.2 and not self.is_dragging:
            self.is_dragging = True
            start_drag()  # Replace with actual function

        # Stop drag if pinch is released
        elif pinch_distance >= 0.2 and self.is_dragging:
            self.is_dragging = False
            stop_drag()  # Replace with actual function

def process_frame(frame_bgr, models, device, hand_features=None, return_metadata=False, detection_threshold=0.5):
    """
    Process a single frame using the models and hand features.

    Args:
        frame_bgr (numpy.ndarray): Input frame in BGR format.
        models (dict): Dictionary containing the initialized models.
        device (torch.device): Device to run the models on.
        hand_features: Object containing hand detection data.

    Returns:
        numpy.ndarray or tuple: Processed frame. If return_metadata=True,
        returns (frame, metadata) where metadata includes detection info.
    """
    start_time = time.time()
    h, w = frame_bgr.shape[:2]
    metadata = {
        "detected": False,
        "score": None,
        "cursor_point": None,
        "keypoints": []
    }

    # STAGE 1: DETECT THE HAND
    img_resized = cv2.resize(frame_bgr, (320, 320))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        stage_one_preds = models['stage_one'](img_tensor)[0]

    if len(stage_one_preds['scores']) == 0:
        if return_metadata:
            return frame_bgr, metadata
        return frame_bgr

    top_score = float(stage_one_preds['scores'][0].item())
    metadata["score"] = top_score
    if top_score < detection_threshold:
        if return_metadata:
            return frame_bgr, metadata
        return frame_bgr

    box = stage_one_preds['boxes'][0].cpu().numpy()
    x_min = int(max(0, box[0] * w / 320))
    y_min = int(max(0, box[1] * h / 320))
    x_max = int(min(w, box[2] * w / 320))
    y_max = int(min(h, box[3] * h / 320))
    cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # STAGE 2: CROP THE HAND & FIND KEYPOINTS
    crop = frame_bgr[y_min:y_max, x_min:x_max]
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return frame_bgr

    crop_h, crop_w = crop.shape[:2]
    crop_resized = cv2.resize(crop, (224, 224))
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

    crop_tensor = torch.tensor(crop_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    crop_tensor = crop_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        heatmaps = models['stage_two'](crop_tensor)[0]

    keypoints = []
    for i in range(heatmaps.shape[0]):
        heatmap = heatmaps[i]
        idx = torch.argmax(heatmap).item()
        y_56 = (idx // heatmap.shape[1])
        x_56 = (idx % heatmap.shape[1])

        y_224 = y_56 * 4
        x_224 = x_56 * 4

        crop_x = int(x_224 * crop_w / 224)
        crop_y = int(y_224 * crop_h / 224)

        actual_x = crop_x + x_min
        actual_y = crop_y + y_min
        keypoints.append((actual_x, actual_y))

        # Draw joints
        if i == 3:
            color, radius = (0, 165, 255), 8 # Orange for Thumb IP Joint
        elif i == 6:
            color, radius = (147, 20, 255), 8 # Pink for Index PIP Joint
        elif i == 9:
            color, radius = (255, 255, 0), 8 # Cyan for Middle Knuckle (Cursor)
        else:
            color, radius = (0, 255, 0), 5
        cv2.circle(frame_bgr, (actual_x, actual_y), radius, color, -1)

    metadata["detected"] = True
    metadata["keypoints"] = keypoints
    if len(keypoints) > 9:
        metadata["cursor_point"] = keypoints[9]

    ai_time_ms = (time.time() - start_time) * 1000
    cv2.putText(frame_bgr, f"AI Time: {ai_time_ms:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if return_metadata:
        return frame_bgr, metadata
    return frame_bgr

def get_hand_position(hand_features):
    """
    Extract the hand position (x, y) from the hand features.

    Args:
        hand_features: Object containing hand detection data.

    Returns:
        tuple: (x, y) coordinates of the hand position.
    """
    return hand_features.palm_x, hand_features.palm_y

    # Check if pinch-to-click is triggered
    if gesture_controller.is_pinch_click(pinch_distance):
        perform_click()  # Replace with actual function

    # Toggle drag functionality
    drag_controller.toggle_drag(pinch_distance)