import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch


def _safe_box(box: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.astype(np.float32)
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    return x1, y1, x2, y2


def process_frame(
    frame_bgr: np.ndarray,
    models: Dict[str, Any],
    device: torch.device,
    return_metadata: bool = False,
    detection_threshold: float = 0.5,
):
    start = time.time()
    frame = frame_bgr.copy()
    orig_h, orig_w = frame.shape[:2]

    stage_one = models["stage_one"]
    stage_two = models["stage_two"]

    resized = cv2.resize(frame, (320, 320))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        det_out = stage_one(tensor)[0]

    scores = det_out.get("scores", torch.tensor([], device=device))
    boxes = det_out.get("boxes", torch.empty((0, 4), device=device))

    score = None
    detected = False
    keypoints: List[Tuple[int, int]] = []
    cursor_point = None

    if scores.numel() > 0:
        top_idx = int(torch.argmax(scores).item())
        score = float(scores[top_idx].item())

        if score >= detection_threshold:
            detected = True
            box = boxes[top_idx].detach().cpu().numpy()
            scale_x = orig_w / 320.0
            scale_y = orig_h / 320.0
            box_scaled = np.array(
                [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y],
                dtype=np.float32,
            )

            x1, y1, x2, y2 = _safe_box(box_scaled, orig_w, orig_h)
            hand_crop = frame[y1:y2, x1:x2]

            if hand_crop.size > 0:
                kp_input = cv2.resize(hand_crop, (224, 224))
                kp_rgb = cv2.cvtColor(kp_input, cv2.COLOR_BGR2RGB)
                kp_tensor = (
                    torch.from_numpy(kp_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                ).to(device)

                with torch.no_grad():
                    heatmaps = stage_two(kp_tensor)[0].detach().cpu().numpy()

                crop_w = max(1, x2 - x1)
                crop_h = max(1, y2 - y1)

                for channel in range(heatmaps.shape[0]):
                    hm = heatmaps[channel]
                    arg_idx = int(np.argmax(hm))
                    hm_h, hm_w = hm.shape
                    hm_y, hm_x = divmod(arg_idx, hm_w)

                    kp_x_224 = hm_x * 4
                    kp_y_224 = hm_y * 4

                    kp_x = int(x1 + (kp_x_224 / 224.0) * crop_w)
                    kp_y = int(y1 + (kp_y_224 / 224.0) * crop_h)

                    keypoints.append((kp_x, kp_y))
                    cv2.circle(frame, (kp_x, kp_y), 3, (0, 255, 0), -1)

                if len(keypoints) > 9:
                    cursor_point = keypoints[9]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

    elapsed_ms = (time.time() - start) * 1000.0
    cv2.putText(
        frame,
        f"AI: {elapsed_ms:.1f}ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    if not return_metadata:
        return frame

    metadata = {
        "detected": detected,
        "score": score,
        "cursor_point": cursor_point,
        "keypoints": keypoints,
    }
    return frame, metadata
