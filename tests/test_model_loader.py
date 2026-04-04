import json
from pathlib import Path

import torch

from model_loader import AirPointerKeypointNet, initialize_models, load_model_weights


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)


def test_load_model_weights(tmp_path: Path):
    model = TinyModel()
    device = torch.device("cpu")

    weights_path = tmp_path / "weights.pth"
    torch.save(model.state_dict(), weights_path)

    ok = load_model_weights(model, str(weights_path), device)
    assert ok is True


def test_airpointer_keypoint_net_shape():
    model = AirPointerKeypointNet()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 21, 56, 56)


def test_initialize_models(tmp_path: Path):
    model_cfg = {
        "custom1": {
            "type": "custom",
            "stage_one": {
                "path": "models/stage_one_model.pth",
                "model_fn": "build_stage_one_detection_model",
            },
            "stage_two": {
                "path": "models/stage_two_model.pth",
                "model_fn": "AirPointerKeypointNet",
            },
        },
        "mediapipe": {"type": "mediapipe", "model_fn": "mediapipe_stub_loader"},
    }

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "models.json"
    cfg_path.write_text(json.dumps(model_cfg), encoding="utf-8")

    models, device = initialize_models(str(cfg_path))

    assert "custom1" in models
    assert "stage_one" in models["custom1"]
    assert "stage_two" in models["custom1"]
    assert "mediapipe" in models
    assert device.type in {"cpu", "cuda"}
