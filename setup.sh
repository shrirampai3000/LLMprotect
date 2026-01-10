#!/bin/bash
# Setup script for Cryptographic Intent Binding project

echo "========================================"
echo "Setting up Anti-LLM Environment"
echo "========================================"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "    source venv/bin/activate"
echo ""
echo "Then you can run:"
echo "    python demo.py           - Run interactive demo"
echo "    python train.py --quick  - Quick training test"
echo "    python -m src.api.server - Start API server"
echo "    python -m pytest tests/  - Run tests"
echo ""
