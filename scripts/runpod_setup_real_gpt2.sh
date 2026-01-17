#!/bin/bash
# RunPod setup script for real GPT-2 experiments
# This script handles dependency conflicts properly

set -e

echo "=== RkCNN-SAE RunPod Setup (Real GPT-2) ==="

# Upgrade pip first
pip install --upgrade pip

# Upgrade PyTorch to 2.6+ (required by transformer-lens)
echo "Upgrading PyTorch to 2.6+..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Install transformer-lens and dependencies
echo "Installing transformer-lens..."
pip install transformer-lens==2.16.1

# Verify transformer-lens
python -c "import transformer_lens; print(f'TransformerLens: {transformer_lens.__version__}')"

# Clone and install RkCNN-SAE
echo "Setting up RkCNN-SAE..."
cd /workspace
if [ -d "RkCNN_SAE" ]; then
    cd RkCNN_SAE && git pull
else
    git clone https://github.com/stchakwdev/RkCNN_SAE.git
    cd RkCNN_SAE
fi

# Install package
pip install -e .

# Create directories
mkdir -p /workspace/results /workspace/checkpoints

# Verify GPU
echo ""
echo "=== Environment Ready ==="
nvidia-smi
python -c "
import torch
import transformer_lens
from rkcnn_sae import RkCNNProbe
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'TransformerLens: {transformer_lens.__version__}')
print('RkCNN-SAE: OK')
print('')
print('Ready to run real GPT-2 experiments!')
"
