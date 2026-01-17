#!/bin/bash
# RunPod Setup Script for RKCNN_SAE
# Run this after creating the pod to set up the environment

set -e  # Exit on error

echo "================================"
echo "RKCNN_SAE RunPod Setup"
echo "================================"

# Navigate to workspace
cd /workspace

# Clone or update repository
if [ -d "RKCNN_SAE" ]; then
    echo "Repository exists, pulling latest..."
    cd RKCNN_SAE
    git pull
else
    echo "Cloning repository..."
    # Replace with your actual repo URL
    git clone https://github.com/YOUR_USERNAME/RKCNN_SAE.git
    cd RKCNN_SAE
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Verify GPU
echo ""
echo "Verifying GPU..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p /workspace/results
mkdir -p /workspace/checkpoints
mkdir -p /workspace/cache

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. Run dry run:    python experiments/phase2_gpt2.py --dry-run"
echo "  2. Run full Phase 2: ./scripts/run_phase2_safe.sh"
echo ""
