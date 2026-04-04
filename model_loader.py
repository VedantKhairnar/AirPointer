import json
from pathlib import Path
from typing import Any, Dict, Tuple

import gdown
import torch
import torch.nn as nn
import torchvision


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
    return torchvision.models.detection.ssdlite320_mobilenet_v3_large(num_classes=2)


def mediapipe_stub_loader() -> str:
    return "mediapipe"


class AirPointerKeypointNet(nn.Module):
    def __init__(self, num_keypoints: int = 21):
        super().__init__()
        self.backbone = torchvision.models.mobilenet_v3_small(weights=None).features
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(576, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        heatmaps = self.decoder(feats)
        heatmaps = torch.nn.functional.interpolate(
            heatmaps, size=(56, 56), mode="bilinear", align_corners=False
        )
        return torch.sigmoid(heatmaps)


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
