# DDW-N2V-2d Environment Guide

This guide explains how to set up and use the Python environment for the DDW-N2V-2d project with CUDA or MPS backend support.

## 🚀 Quick Start

### 1. Setup the Environment

Run the setup script to create a virtual environment and install all dependencies:

```bash
cd /path/to/DDW-N2V-2d
./setup_environment.sh
```

This script will:
- Create a virtual environment using `uv`
- Install PyTorch with CUDA support (if CUDA toolkit is available)
- Install PyTorch with MPS support (on macOS)
- Install all required dependencies

### 2. Activate the Environment

```bash
source .venv/bin/activate
```

### 3. Check Your Environment

```bash
python check_environment.py
```

This will verify all dependencies and check CUDA/MPS availability.

### 4. Run Training

You can run training in several ways:

#### Option A: Using the run script
```bash
./run_training.sh
```

#### Option B: Direct command with parameters
```bash
python src/ddw/fit_n2v_model.py \
    --unet-params-dict "{'chans': 64, 'num_downsample_layers': 3, 'drop_prob': 0.3}" \
    --adam-params-dict "{'lr': 4e-2}" \
    --num-epochs 400 \
    --batch-size 32 \
    --gpu 0 \
    --subtomo-size 128 \
    --mw-angle 50 \
    --subtomo-dir "path/to/subtomos" \
    --project-dir "path/to/project" \
    --n2v-masked-pixel-percentage 0.2 \
    --n2v-roi-size 11 \
    --n2v-strategy "uniform"
```

#### Option C: Using a configuration file
```bash
# Edit config_example.yaml with your parameters
python src/ddw/fit_n2v_model.py --yaml-config config_example.yaml
```

## 📋 Environment Configuration

### Project Files

- `pyproject.toml` - Main project configuration with dependencies
- `requirements.txt` - Alternative dependency specification
- `setup_environment.sh` - Environment setup script
- `check_environment.py` - Environment verification script
- `run_training.sh` - Training execution script
- `config_example.yaml` - Example configuration file

### Dependencies

The project requires the following main dependencies:

- **PyTorch** (>= 2.2.0) - Deep learning framework
- **PyTorch Lightning** (>= 2.2.0) - Training framework
- **NumPy** (>= 1.24.0) - Numerical computing
- **SciPy** (>= 1.10.0) - Scientific computing
- **Matplotlib** (>= 3.7.0) - Visualization
- **MRCFile** (>= 1.4.0) - MRC file support
- **StarFile** (>= 0.5.0) - STAR file support
- **scikit-image** (>= 0.20.0) - Image processing
- **Accelerate** (>= 0.27.0) - Training acceleration

## 🎯 GPU Acceleration

The environment automatically detects and configures GPU acceleration:

### CUDA Support (NVIDIA GPUs)

If the CUDA toolkit is installed, the setup script will install PyTorch with CUDA support.

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- NVIDIA drivers installed

**Verification:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Usage in training:**
```bash
# Use first GPU
python src/ddw/fit_n2v_model.py --gpu 0 ...

# Use multiple GPUs
python src/ddw/fit_n2v_model.py --gpu [0,1] ...
```

### MPS Support (Apple Silicon)

On macOS with Apple Silicon, the setup script will install PyTorch with MPS (Metal Performance Shaders) support.

**Requirements:**
- macOS 12.3+ (Monterey) or later
- Apple Silicon (M1, M2, etc.)

**Verification:**
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Usage in training:**
```bash
# PyTorch Lightning will automatically use MPS
python src/ddw/fit_n2v_model.py --accelerator mps ...
```

### CPU Fallback

If no GPU acceleration is available, the environment will fall back to CPU-only PyTorch.

## 🔧 Manual Environment Management

### Using uv Directly

You can use `uv` directly for environment management:

```bash
# Create environment
uv venv .venv

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install specific version of PyTorch with CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install development dependencies
uv pip install pytest ruff mypy jupyter
```

### Installing Specific PyTorch Versions

#### CUDA 12.1
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CUDA 11.8
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CPU-only
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### MPS (macOS)
```bash
uv pip install torch torchvision torchaudio
```

## 📊 Environment Verification

Run the environment check script to verify everything is working:

```bash
python check_environment.py
```

This script will check:
- Python version
- PyTorch installation and GPU support
- All required dependencies
- Project structure
- N2V module imports
- Basic N2V functionality

## 🔄 Updating Dependencies

To update dependencies:

```bash
# Activate environment
source .venv/bin/activate

# Update all packages
uv pip install --upgrade -r requirements.txt

# Or update specific packages
uv pip install --upgrade torch pytorch-lightning
```

## 🧹 Cleaning Up

To remove the virtual environment:

```bash
rm -rf .venv
```

## 🐛 Troubleshooting

### Common Issues

#### CUDA not detected

1. Verify CUDA toolkit is installed:
   ```bash
   nvcc --version
   ```

2. Check NVIDIA drivers:
   ```bash
   nvidia-smi
   ```

3. Reinstall PyTorch with CUDA:
   ```bash
   uv pip uninstall torch torchvision torchaudio
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

#### MPS not available on macOS

1. Ensure you have macOS 12.3+ (Monterey) or later
2. Ensure you're using Apple Silicon (M1, M2, etc.)
3. Update to the latest PyTorch version

#### Import errors

If you get import errors, ensure the virtual environment is activated:

```bash
source .venv/bin/activate
```

#### Permission errors

If you get permission errors when installing packages, try:

```bash
uv pip install --user package_name
```

Or use `--break-system-packages` if needed:

```bash
uv pip install --break-system-packages package_name
```

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)

## 🎓 Best Practices

1. **Always use the virtual environment** when working with the project
2. **Regularly update dependencies** to get the latest features and bug fixes
3. **Use configuration files** for complex training setups
4. **Monitor GPU usage** during training with `nvidia-smi` (CUDA) or Activity Monitor (MPS)
5. **Start with small batch sizes** when testing new configurations
6. **Use mixed precision** (`precision=16`) for faster training when possible

## 🔬 Performance Optimization

### CUDA Optimization

- Use `pin_memory=True` in DataLoader for faster GPU transfers
- Use `persistent_workers=True` to maintain workers between epochs
- Use mixed precision training (`precision=16`)
- Use multiple GPUs with `gpu=[0,1,2]`

### MPS Optimization

- Use smaller batch sizes (MPS has memory limitations)
- Use `accelerator='mps'` in PyTorch Lightning
- Monitor memory usage in Activity Monitor

### General Optimization

- Use `num_workers` equal to your CPU core count
- Use `shuffle=True` for better training
- Use appropriate `batch_size` based on your GPU memory
- Use `n2v_strategy="uniform"` for faster training (vs "median")
