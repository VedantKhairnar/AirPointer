import numpy as np
import torch

from inference_pipeline import process_frame


class DummyStageOne:
    def __call__(self, x):
        device = x.device
        boxes = torch.tensor([[40.0, 50.0, 250.0, 280.0]], device=device)
        scores = torch.tensor([0.9], device=device)
        return [{"boxes": boxes, "scores": scores}]


class DummyStageTwo:
    def __call__(self, x):
        heatmaps = torch.zeros((1, 21, 56, 56), device=x.device)
        for i in range(21):
            heatmaps[0, i, min(55, i + 5), min(55, i + 7)] = 1.0
        return heatmaps


def test_process_frame_with_metadata():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    models = {"stage_one": DummyStageOne(), "stage_two": DummyStageTwo()}
    device = torch.device("cpu")

    out, metadata = process_frame(
        frame,
        models,
        device,
        return_metadata=True,
        detection_threshold=0.5,
    )

    assert out.shape == frame.shape
    assert metadata["detected"] is True
    assert metadata["score"] is not None
    assert len(metadata["keypoints"]) == 21
    assert metadata["cursor_point"] is not None


def test_process_frame_below_threshold():
    class LowScoreStageOne:
        def __call__(self, x):
            device = x.device
            boxes = torch.tensor([[20.0, 20.0, 100.0, 100.0]], device=device)
            scores = torch.tensor([0.1], device=device)
            return [{"boxes": boxes, "scores": scores}]

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    models = {"stage_one": LowScoreStageOne(), "stage_two": DummyStageTwo()}

    out, metadata = process_frame(
        frame,
        models,
        torch.device("cpu"),
        return_metadata=True,
        detection_threshold=0.5,
    )

    assert out.shape == frame.shape
    assert metadata["detected"] is False
    assert metadata["cursor_point"] is None
    assert metadata["keypoints"] == []


def test_process_frame_skips_stage_one_on_track_frames():
    class CountingStageOne:
        def __init__(self):
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            device = x.device
            boxes = torch.tensor([[40.0, 50.0, 250.0, 280.0]], device=device)
            scores = torch.tensor([0.9], device=device)
            return [{"boxes": boxes, "scores": scores}]

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    stage_one = CountingStageOne()
    models = {"stage_one": stage_one, "stage_two": DummyStageTwo()}
    device = torch.device("cpu")

    _, first = process_frame(
        frame,
        models,
        device,
        return_metadata=True,
        detection_threshold=0.5,
        frame_index=1,
        detect_interval=5,
        last_bbox=None,
    )

    assert first["detected"] is True
    assert stage_one.calls == 1

    _, second = process_frame(
        frame,
        models,
        device,
        return_metadata=True,
        detection_threshold=0.5,
        frame_index=2,
        detect_interval=5,
        last_bbox=np.array(first["bbox"], dtype=np.float32),
    )

    assert second["detected"] is True
    assert second["stage_one_ran"] is False
    assert stage_one.calls == 1
