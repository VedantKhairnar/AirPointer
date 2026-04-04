# Model Assets

This directory contains the runtime model weights and assets required by AirPointer.

## Runtime Assets

- **hand_landmarker.task** — MediaPipe hand landmark detection model (tasks API format)
- **stage_one_model.pth** — Custom hand detection backbone (MobileNetV3-Small + SSD)
- **stage_two_model.pth** — Custom keypoint regression backbone (MobileNetV3-Small decoder)

## Reference & Setup Files

The following reference and helper files have been moved to [../temp/](../temp/) for organization:

- **temp/MODEL_REFERENCE.md** — Technical architecture reference for the two-stage pipeline
- **temp/run_live.py** — Standalone inference script (alternative to main_advanced.py)
- **temp/download_hand_landmark_model.sh** — Shell script to fetch MediaPipe model
- **temp/live_run_project.ipynb** — Jupyter notebook for interactive testing
- **temp/requirements.txt** — Dependencies for reference/helper scripts

## How to use

### Standard runtime

```bash
# From project root
python main_advanced.py --model custom1     # Two-stage custom model
python main_advanced.py --model mediapipe   # MediaPipe single-stage
```

### Reference/standalone testing

See [../temp/MODEL_REFERENCE.md](../temp/MODEL_REFERENCE.md) for architecture details.

For standalone custom inference outside the main framework:
```bash
cd temp/
python run_live.py
```

For interactive notebook-based testing:
```bash
cd temp/
jupyter notebook live_run_project.ipynb
```

To download fresh MediaPipe model:
```bash
cd ..
bash temp/download_hand_landmark_model.sh
```
