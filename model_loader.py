import json
from pathlib import Path
from typing import Any, Dict, Tuple

import gdown
import torch
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v3_small
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD


def download_weights(file_id: str, output_path: str) -> bool:
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception:
        return False


def load_model_weights(model: nn.Module, weights_path: str, device: torch.device) -> bool:
    path = Path(weights_path)
    if not path.exists():
        return False

    try:
        state = torch.load(str(path), map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        return True
    except Exception:
        return False


def build_stage_one_detection_model() -> nn.Module:
    """
    Stage 1 — Hand detector.
    MobileNetV3-Small backbone + SSD head.
    Input : (1, 3, 320, 320)  float32, values in [0, 1]
    Output: list of dicts with keys 'boxes' (N,4) and 'scores' (N,)
    """
    backbone = mobilenet_v3_small(weights=None).features
    backbone.out_channels = [576]
    box_gen = DefaultBoxGenerator(
        aspect_ratios=[[2, 3]], min_ratio=0.2, max_ratio=0.95
    )
    return SSD(backbone, num_classes=2, anchor_generator=box_gen, size=(320, 320))


def mediapipe_stub_loader() -> str:
    return "mediapipe"


class AirPointerKeypointNet(nn.Module):
    """
    Stage 2 — Keypoint detector.
    MobileNetV3-Small features (576-ch @ 7x7) decoded to 21-channel
    heatmaps at 56x56 resolution.
    Output is passed through Sigmoid, so values are in [0, 1].

    Input : (1, 3, 224, 224)  float32, values in [0, 1]
    Output: (1, 21, 56, 56)   float32, heatmap per keypoint
    """
    def __init__(self):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=None).features
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(576, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 21, kernel_size=1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        # Manually upsample since decoder doesn't have intermediate upsamples
        x = nn.functional.interpolate(feats, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decoder(x)
        # Final upsampling to 56x56
        x = nn.functional.interpolate(x, size=(56, 56), mode="bilinear", align_corners=False)
        return self.sigmoid(x)


def initialize_models(config_path: str) -> Tuple[Dict[str, Any], torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_file = Path(config_path)
    repo_root = config_file.parent.parent.resolve()

    with config_file.open("r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    models: Dict[str, Any] = {}

    if "custom1" in model_cfg:
        custom_cfg = model_cfg["custom1"]

        stage_one = build_stage_one_detection_model().to(device)
        stage_two = AirPointerKeypointNet().to(device)

        stage_one_path = str((repo_root / custom_cfg["stage_one"]["path"]).resolve())
        stage_two_path = str((repo_root / custom_cfg["stage_two"]["path"]).resolve())

        stage_one_loaded = load_model_weights(stage_one, stage_one_path, device)
        stage_two_loaded = load_model_weights(stage_two, stage_two_path, device)

        stage_one.eval()
        stage_two.eval()

        models["custom1"] = {
            "stage_one": stage_one,
            "stage_two": stage_two,
            "stage_one_loaded": stage_one_loaded,
            "stage_two_loaded": stage_two_loaded,
            "stage_one_path": stage_one_path,
            "stage_two_path": stage_two_path,
        }

    if "mediapipe" in model_cfg:
        models["mediapipe"] = mediapipe_stub_loader()

    return models, device
