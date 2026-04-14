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
    keypoint_peak_threshold: float = 0.2,
    frame_index: int | None = None,
    detect_interval: int = 1,
    last_bbox: np.ndarray | None = None,
    force_stage_one: bool = False,
):
    start = time.time()
    frame = frame_bgr.copy()
    orig_h, orig_w = frame.shape[:2]

    stage_one = models["stage_one"]
    stage_two = models["stage_two"]

    should_run_stage_one = force_stage_one or last_bbox is None
    if (
        not should_run_stage_one
        and last_bbox is not None
        and detect_interval > 1
        and frame_index is not None
    ):
        should_run_stage_one = frame_index % detect_interval == 0

    request_stage_one_next = False

    def _run_stage_two_from_bbox(bbox: np.ndarray):
        x1, y1, x2, y2 = [int(v) for v in bbox.tolist()]
        hand_crop = frame[y1:y2, x1:x2]

        if hand_crop.size == 0:
            return False, [], None, None

        kp_input = cv2.resize(hand_crop, (224, 224))
        kp_rgb = cv2.cvtColor(kp_input, cv2.COLOR_BGR2RGB)
        kp_tensor = (
            torch.from_numpy(kp_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        ).to(device)

        with torch.no_grad():
            heatmaps = stage_two(kp_tensor)[0].detach().cpu().numpy()

        crop_w = max(1, x2 - x1)
        crop_h = max(1, y2 - y1)
        peak_scores: List[float] = []
        keypoints_local: List[Tuple[int, int]] = []

        for channel in range(heatmaps.shape[0]):
            hm = heatmaps[channel]
            arg_idx = int(np.argmax(hm))
            hm_h, hm_w = hm.shape
            hm_y, hm_x = divmod(arg_idx, hm_w)
            peak_scores.append(float(hm[hm_y, hm_x]))

            kp_x_224 = hm_x * 4
            kp_y_224 = hm_y * 4

            kp_x = int(x1 + (kp_x_224 / 224.0) * crop_w)
            kp_y = int(y1 + (kp_y_224 / 224.0) * crop_h)
            keypoints_local.append((kp_x, kp_y))

        mean_peak_local = float(np.mean(peak_scores)) if peak_scores else None
        if mean_peak_local is None or mean_peak_local < keypoint_peak_threshold:
            return False, [], None, mean_peak_local

        cursor_local = keypoints_local[9] if len(keypoints_local) > 9 else None
        return True, keypoints_local, cursor_local, mean_peak_local

    scores = torch.tensor([], device=device)
    boxes = torch.empty((0, 4), device=device)

    if should_run_stage_one or last_bbox is None:
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
    mean_peak_score = None

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
            last_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            detected, keypoints, cursor_point, mean_peak_score = _run_stage_two_from_bbox(last_bbox)
            if detected:
                for kp_x, kp_y in keypoints:
                    cv2.circle(frame, (kp_x, kp_y), 3, (0, 255, 0), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            else:
                request_stage_one_next = True
        else:
            request_stage_one_next = True

    if scores.numel() == 0 and should_run_stage_one:
        request_stage_one_next = True

    if not detected and last_bbox is not None and not should_run_stage_one:
        x1, y1, x2, y2 = [int(v) for v in last_bbox.tolist()]
        detected, keypoints, cursor_point, mean_peak_score = _run_stage_two_from_bbox(last_bbox)
        if detected:
            for kp_x, kp_y in keypoints:
                cv2.circle(frame, (kp_x, kp_y), 3, (0, 255, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        else:
            request_stage_one_next = True

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
        "mean_peak_score": mean_peak_score,
        "cursor_point": cursor_point,
        "keypoints": keypoints,
        "stage_one_ran": should_run_stage_one or last_bbox is None,
        "bbox": [int(v) for v in last_bbox.tolist()] if last_bbox is not None else None,
        "request_stage_one_next": request_stage_one_next,
    }
    return frame, metadata
