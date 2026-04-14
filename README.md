# AirPointer

AirPointer is a hand-tracking mouse control app with two model modes:

- `mediapipe` (fast and lightweight)
- `custom1` (two-stage custom detector + keypoints)

It supports cursor movement, click, drag, and open-hand app switching on macOS.

## Project Structure

- `main_advanced.py` - Main runtime used by the app
- `config.py` - Runtime thresholds and behavior flags
- `model_loader.py` - Custom model architecture and weight loading
- `inference_pipeline.py` - Custom model inference pipeline
- `electron-ui/` - Desktop UI and backend process manager
- `models/` - Runtime model files (`.pth`, `.task`)

## Prerequisites

- macOS (tested)
- Python virtual environment at:
  - `/Users/vedantkhairnar/Documents/RIT/DL/FinalProject/.venv`
- Node.js + npm (for Electron UI)
- Webcam access enabled

## Python Setup

From repository parent:

```bash
cd /Users/vedantkhairnar/Documents/RIT/DL/FinalProject
source .venv/bin/activate
```

If dependencies are missing:

```bash
cd /Users/vedantkhairnar/Documents/RIT/DL/FinalProject/Airpointer
pip install -r requirements.txt
```

## Run The Application (CLI)

From `Airpointer/`:

```bash
cd /Users/vedantkhairnar/Documents/RIT/DL/FinalProject/Airpointer
source ../.venv/bin/activate
python main_advanced.py --model mediapipe
```

Run with custom model:

```bash
python main_advanced.py --model custom1
```

You can also use the helper script:

```bash
./run.sh mediapipe
# or
./run.sh custom1
```

## Run The Application (Electron UI)

From `electron-ui/`:

```bash
cd /Users/vedantkhairnar/Documents/RIT/DL/FinalProject/Airpointer/electron-ui
npm install
npm start
```

The UI launches the Python backend automatically. You can switch between `mediapipe` and `custom1` from the model dropdown.

## Controls

When running in non-embedded OpenCV window mode:

- `q` - quit
- `d` - toggle debug overlays
- `m` - toggle mouse actions

## Logs

- Main runtime logs: `application.log`
- Electron backend logs are also exposed in the UI and temp log files.

## Tests

From `Airpointer/`:

```bash
cd /Users/vedantkhairnar/Documents/RIT/DL/FinalProject/Airpointer
source ../.venv/bin/activate
PYTHONPATH=. pytest -q
```

## Notes

- `custom1` uses model files from `models/stage_one_model.pth` and `models/stage_two_model.pth`.
- `mediapipe` uses `models/hand_landmarker.task`.
