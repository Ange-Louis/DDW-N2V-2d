#!/bin/bash

# Script to run DDW-N2V-2d training with the correct environment

set -e

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run setup_environment.sh first."
    exit 1
fi

# Activate the virtual environment
source .venv/bin/activate

echo "Environment activated. Running training..."
echo ""

# Check if CUDA is available
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA is available - training will use GPU"
    python -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}')"
    python -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
elif python -c "import torch; print(torch.backends.mps.is_available())" | grep -q "True"; then
    echo "MPS (Apple Metal) is available - training will use MPS"
else
    echo "No GPU acceleration available - training will use CPU"
fi

echo ""
echo "Starting training..."
echo ""

# Run the training script with example parameters
# You can modify this or pass your own parameters
python src/ddw/fit_n2v_model.py \
    --unet-params-dict "{'chans': 64, 'num_downsample_layers': 3, 'drop_prob': 0.3}" \
    --adam-params-dict "{'lr': 4e-2}" \
    --num-epochs 10 \
    --batch-size 8 \
    --num-workers 4 \
    --gpu 0 \
    --subtomo-size 128 \
    --mw-angle 50 \
    --subtomo-dir "example_data/subtomos" \
    --project-dir "example_project" \
    --n2v-masked-pixel-percentage 0.2 \
    --n2v-roi-size 11 \
    --n2v-strategy "uniform"

# To use a configuration file instead:
# python src/ddw/fit_n2v_model.py --yaml-config config_example.yaml
