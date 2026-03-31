import cv2
import time
import torch
import numpy as np

def process_frame(frame_bgr, models, device):
    """
    Process a single frame using the models.

    Args:
        frame_bgr (numpy.ndarray): Input frame in BGR format.
        models (dict): Dictionary containing the initialized models.
        device (torch.device): Device to run the models on.

    Returns:
        numpy.ndarray: Processed frame with annotations.
    """
    start_time = time.time()
    h, w = frame_bgr.shape[:2]

    # STAGE 1: DETECT THE HAND
    img_resized = cv2.resize(frame_bgr, (320, 320))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        stage_one_preds = models['stage_one'](img_tensor)[0]

    if len(stage_one_preds['scores']) == 0 or stage_one_preds['scores'][0] < 0.5:
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

    ai_time_ms = (time.time() - start_time) * 1000
    cv2.putText(frame_bgr, f"AI Time: {ai_time_ms:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame_bgr