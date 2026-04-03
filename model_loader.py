import os
import torch
import gdown
import json
import logging
from torchvision.models import mobilenet_v3_small
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
import torch.nn as nn

# --- UTILITY FUNCTIONS ---
def download_weights(file_id, output_path):
    if not os.path.exists(output_path):
        gdown.download(id=file_id, output=output_path, quiet=False)

def load_model_weights(model, weights_path, device):
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"Weights loaded for {weights_path}")
        return True
    except Exception as e:
        logging.error(f"Error loading weights for {weights_path}: {e}")
        return False

# --- MODEL DEFINITIONS ---
def build_stage_one_detection_model():
    feature_extractor = mobilenet_v3_small(weights='DEFAULT').features
    feature_extractor.out_channels = [576]
    box_generator = DefaultBoxGenerator(aspect_ratios=[[2, 3]], min_ratio=0.2, max_ratio=0.95)
    return SSD(feature_extractor, num_classes=2, anchor_generator=box_generator, size=(320, 320))

class AirPointerKeypointNet(nn.Module):
    def __init__(self):
        super(AirPointerKeypointNet, self).__init__()
        self.backbone = mobilenet_v3_small(weights='DEFAULT').features
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(576, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 21, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.backbone(x)
        heatmaps = self.decoder(features)
        return self.sigmoid(heatmaps)

# --- MEDIAPIPE STUB LOADER ---
def mediapipe_stub_loader():
    """Stub loader for Mediapipe model. Returns a dummy object or None."""
    print("Mediapipe model selected (stub). No weights loaded.")
    return None

# --- MODEL LOADER ---
def initialize_models(config_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    config_dir = os.path.dirname(os.path.abspath(config_path))
    repo_root = os.path.abspath(os.path.join(config_dir, os.pardir))

    def resolve_model_path(model_path):
        if os.path.isabs(model_path):
            return model_path
        return os.path.abspath(os.path.join(repo_root, model_path))

    # Load configuration from JSON
    with open(config_path, "r") as f:
        config = json.load(f)

    logging.info(f"Model config loaded from {config_path}: {list(config.keys())}")

    for key, entry in config.items():
        if key == "custom1":
            logging.info("Initializing custom1 pipeline...")
            # Load both stage_one and stage_two as part of custom1
            stage_one_cfg = entry["stage_one"]
            stage_two_cfg = entry["stage_two"]
            model_one_fn = globals()[stage_one_cfg["model_fn"]]
            model_two_fn = globals()[stage_two_cfg["model_fn"]]
            stage_one_path = resolve_model_path(stage_one_cfg["path"])
            stage_two_path = resolve_model_path(stage_two_cfg["path"])
            logging.info(f"Loading stage_one model from {stage_one_path} using {stage_one_cfg['model_fn']}")
            logging.info(f"Loading stage_two model from {stage_two_path} using {stage_two_cfg['model_fn']}")
            model_one = model_one_fn()
            model_two = model_two_fn()
            if not os.path.exists(stage_one_path):
                logging.error(f"Stage one weights not found: {stage_one_path}")
            else:
                load_model_weights(model_one, stage_one_path, device)
            if not os.path.exists(stage_two_path):
                logging.error(f"Stage two weights not found: {stage_two_path}")
            else:
                load_model_weights(model_two, stage_two_path, device)
            model_one.to(device).eval()
            model_two.to(device).eval()
            models["custom1"] = {"stage_one": model_one, "stage_two": model_two}
            logging.info("Custom1 pipeline models loaded and moved to device.")
        elif key == "mediapipe":
            logging.info("Initializing mediapipe model...")
            model_fn = globals()[entry["model_fn"]]
            model = model_fn()
            if model is not None:
                model.to(device).eval()
            models["mediapipe"] = model
            logging.info("Mediapipe model loaded.")
        else:
            logging.warning(f"Unknown model key '{key}' in config. Skipping.")

    logging.info("All models initialized successfully!")
    return models, device

# --- MODEL MANAGER ---
class ModelManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.models = {}
        self.current_model = None
        self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as f:
            self.models = json.load(f)  # Directly load the existing JS-style config

    def switch_model(self, stage):
        if stage not in self.models:
            raise ValueError(f"Stage '{stage}' not found in configuration.")

        model_config = self.models[stage]
        model_fn = globals()[model_config['model_fn']]
        model = model_fn()
        self.current_model = model
        print(f"Switched to model: {stage}")
