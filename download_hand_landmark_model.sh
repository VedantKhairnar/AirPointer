#!/bin/bash
# Download MediaPipe hand landmark model for tasks API
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task/hand_landmarker.task"
MODEL_PATH="hand_landmarker.task"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading hand landmark model..."
    curl -L "$MODEL_URL" -o "$MODEL_PATH"
    echo "Model downloaded to $MODEL_PATH"
else
    echo "Model already exists at $MODEL_PATH"
fi
