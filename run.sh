#!/bin/bash

echo "🚀 Drug Addiction Detection System"
echo "=================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if packages are installed
echo "📦 Checking packages..."
python -c "import cv2, numpy, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing required packages..."
    pip install -r requirements.txt
fi

# Test camera first
echo "📹 Testing camera access..."
python test_camera.py
if [ $? -ne 0 ]; then
    echo "❌ Camera test failed!"
    exit 1
fi

# Check if model exists, if not train it
if [ ! -f "drug_detection_model.pkl" ]; then
    echo "🧠 Training detection model..."
    python train_model.py
fi

# Run the main application
echo "🏃 Starting Drug Detection System..."
echo "Press 'q' in the camera window to quit"
python main.py

echo "👋 Goodbye!"
