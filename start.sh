#!/bin/bash

# AirPointer Quick Start Script
# This script sets up and runs AirPointer

echo "=========================================="
echo "AirPointer - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"
echo ""

# Check for virtual environment
if [ -d "venv" ]; then
    echo "[2/5] Virtual environment found, activating..."
    source venv/bin/activate
else
    echo "[2/5] Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "  ✓ Virtual environment created"
fi
echo ""

# Install dependencies
echo "[3/5] Installing dependencies..."
pip install -r requirements.txt
echo "  ✓ Dependencies installed"
echo ""

# Download model if missing
if [ ! -f "hand_landmarker.task" ]; then
    echo "[4/5] Downloading MediaPipe hand landmark model (7.5 MB)..."
    curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o hand_landmarker.task
    if [ -f "hand_landmarker.task" ] && [ $(stat -f%z "hand_landmarker.task" 2>/dev/null || stat -c%s "hand_landmarker.task" 2>/dev/null) -gt 1000000 ]; then
        echo "  ✓ Model downloaded successfully"
    else
        echo "  ✗ Model download failed. Please download manually from:"
        echo "    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        exit 1
    fi
else
    echo "[4/5] MediaPipe model found"
fi
echo ""

# Run tests
echo "[5/5] Running system tests..."
python test.py
if [ $? -ne 0 ]; then
    echo "  ✗ Tests failed. Please check the output above."
    exit 1
fi
echo ""

# Start AirPointer
echo "Starting AirPointer..."
echo ""
echo "System starting. Please ensure:"
echo "  • Webcam is connected and working"
echo "  • You have good lighting"
echo "  • You're in a position where your hand is visible"
echo ""
echo "Press 'm' to enable mouse control (starts disabled for safety)"
echo ""
python main.py
