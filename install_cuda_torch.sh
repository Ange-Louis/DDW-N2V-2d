#!/bin/bash

# Script to install PyTorch with CUDA support

set -e

echo "Checking for NVIDIA GPU and CUDA toolkit..."

# Check if NVIDIA drivers are loaded
if lspci | grep -i nvidia; then
    echo "✓ NVIDIA GPU detected"
    
    # Check if CUDA toolkit is installed
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d. -f1-2)
        echo "✓ CUDA toolkit version: $CUDA_VERSION"
        
        # Determine appropriate PyTorch CUDA version
        if [[ "$CUDA_VERSION" == "12.1" ]]; then
            CUDA_SUFFIX="cu121"
        elif [[ "$CUDA_VERSION" == "12.4" ]]; then
            CUDA_SUFFIX="cu124"
        elif [[ "$CUDA_VERSION" == "11.8" ]]; then
            CUDA_SUFFIX="cu118"
        elif [[ "$CUDA_VERSION" == "11.7" ]]; then
            CUDA_SUFFIX="cu117"
        else
            echo "CUDA version $CUDA_VERSION not directly supported, using cu121"
            CUDA_SUFFIX="cu121"
        fi
        
        echo "Installing PyTorch with CUDA $CUDA_SUFFIX support..."
        
        # Activate environment
        source .venv/bin/activate
        
        # Uninstall CPU-only PyTorch
        uv pip uninstall torch torchvision torchaudio -y
        
        # Install CUDA PyTorch
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$CUDA_SUFFIX
        
        echo "✓ PyTorch with CUDA support installed"
        
        # Verify
        python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
        python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
        python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
        
    else
        echo "✗ CUDA toolkit not found. Please install it first."
        echo "  Download from: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
else
    echo "✗ No NVIDIA GPU detected"
    exit 1
fi
