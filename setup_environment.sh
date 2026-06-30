#!/bin/bash

# DDW-N2V-2d Environment Setup Script
# This script sets up a Python environment with uv for CUDA/MPS support

set -e

echo "=========================================="
echo "DDW-N2V-2d Environment Setup"
echo "=========================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv found: $(uv --version)"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate environment and install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Try to install PyTorch with CUDA if available
echo "Checking for CUDA support..."
if command -v nvcc &> /dev/null || [ -f /usr/local/cuda/bin/nvcc ]; then
    echo "CUDA toolkit found, installing PyTorch with CUDA support..."
    uv pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo "✓ PyTorch with CUDA installed"
elif [ "$(uname)" == "Darwin" ]; then
    echo "macOS detected, installing PyTorch with MPS support..."
    uv pip install --upgrade --force-reinstall torch torchvision torchaudio
    echo "✓ PyTorch with MPS installed"
else
    echo "No CUDA toolkit found, installing CPU-only PyTorch..."
    uv pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo "✓ PyTorch CPU-only installed"
fi

# Install remaining dependencies
echo "Installing additional dependencies..."
uv pip install pytorch-lightning typer typer-config numpy scipy tqdm matplotlib mrcfile starfile scikit-image accelerate

echo ""
echo "=========================================="
echo "Environment Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To check CUDA/MPS availability:"
echo "  python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')\""
echo ""
