import os
import torch
import gdown
import json
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
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Weights loaded for {weights_path}")
    except Exception as e:
        print(f"Error loading weights for {weights_path}: {e}")

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

# --- MODEL LOADER ---
def initialize_models(config_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}

    # Load configuration from JSON
    with open(config_path, "r") as f:
        config = json.load(f)

    for stage, stage_config in config.items():
        print(f"Initializing {stage} model...")
        model_fn = globals()[stage_config['model_fn']]
        model = model_fn()
        
        # Only download and load weights if they exist in config (not for mediapipe)
        if 'file_id' in stage_config and 'weights_path' in stage_config:
            download_weights(stage_config['file_id'], stage_config['weights_path'])
            load_model_weights(model, stage_config['weights_path'], device)
        
        # Always set to eval mode to avoid training mode assertions
        model.to(device).eval()
        models[stage] = model

    print("All models initialized successfully!")
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

# Example usage
model_manager = ModelManager("config/models.json")
model_manager.switch_model("stage_one")