# DDW-N2V-2d Project

A PyTorch-based implementation of Noise2Void (N2V) for denoising and missing wedge reconstruction on sub-tomograms, with support for CUDA and MPS backends.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository (if not already done)
git clone https://github.com/Ange-Louis/DDW-N2V-2d.git
cd DDW-N2V-2d

# Setup the Python environment with uv
./setup_environment.sh
```

### 2. Activate Environment

```bash
source .venv/bin/activate
```

### 3. Check Environment

```bash
python check_environment.py
```

### 4. Run Training

```bash
# Using the run script
./run_training.sh

# Or directly with parameters
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

# Or using a configuration file
python src/ddw/fit_n2v_model.py --yaml-config config_example.yaml
```

## 📋 Project Structure

```
DDW-N2V-2d/
├── src/
│   └── ddw/
│       ├── fit_n2v_model.py          # Main training script
│       ├── prepare_data2.py          # Data preparation
│       └── utils/
│           ├── n2v_utils.py          # Core N2V utilities
│           ├── n2v_subtomo_dataset.py # N2V dataset implementation
│           ├── n2v_unet2.py          # N2V U-Net model
│           ├── fourier2.py           # Fourier transform utilities
│           ├── missing_wedge2.py     # Missing wedge handling
│           ├── rotation2.py          # Rotation utilities
│           └── ...
├── pyproject.toml                   # Project configuration
├── requirements.txt                 # Dependencies
├── setup_environment.sh             # Environment setup script
├── check_environment.py             # Environment verification
├── run_training.sh                  # Training execution script
├── install_cuda_torch.sh            # CUDA PyTorch installation
├── config_example.yaml              # Example configuration
├── ENVIRONMENT_GUIDE.md              # Detailed environment guide
├── N2V_IMPLEMENTATION.md            # N2V implementation details
└── README.md                        # This file
```

## 🎯 Features

- **Noise2Void Implementation**: Self-supervised denoising from single noisy images
- **Noise2Void2 Support**: Median strategy for more robust denoising
- **Missing Wedge Reconstruction**: Handle missing wedge artifacts in tomography
- **Multi-GPU Support**: Train on multiple GPUs
- **CUDA Acceleration**: Full CUDA support for NVIDIA GPUs
- **MPS Acceleration**: Apple Metal support for macOS
- **PyTorch Lightning**: Modern training framework with built-in best practices

## 📦 Dependencies

### Core Dependencies
- Python >= 3.10
- PyTorch >= 2.2.0
- PyTorch Lightning >= 2.2.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- MRCFile >= 1.4.0
- StarFile >= 0.5.0
- scikit-image >= 0.20.0
- Accelerate >= 0.27.0

### Development Dependencies (Optional)
- pytest >= 7.4.0
- ruff >= 0.1.0
- mypy >= 1.8.0
- Jupyter >= 1.0.0

## 🎨 N2V Strategies

### Uniform Strategy (Original N2V)
- Replaces masked pixels with values from neighboring pixels
- Uses uniform random selection from a local region
- Faster computation
- Configured with `n2v_strategy="uniform"`

### Median Strategy (N2V2)
- Replaces masked pixels with the median of their neighborhood
- More robust to outliers
- Slightly slower computation
- Configured with `n2v_strategy="median"`

## 🔧 Configuration

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unet_params_dict` | dict | - | U-Net model parameters |
| `adam_params_dict` | dict | - | Adam optimizer parameters |
| `num_epochs` | int | 400 | Number of training epochs |
| `batch_size` | int | 32 | Batch size for training |
| `gpu` | int/list | 0 | GPU device(s) to use |
| `subtomo_size` | int | 128 | Size of subtomograms |
| `mw_angle` | float | 50 | Missing wedge angle |
| `subtomo_dir` | str | - | Directory containing subtomograms |
| `project_dir` | str | - | Directory for project outputs |
| `n2v_masked_pixel_percentage` | float | 0.2 | Percentage of pixels to mask |
| `n2v_roi_size` | int | 11 | Region size for pixel replacement |
| `n2v_strategy` | str | "uniform" | Strategy: "uniform" or "median" |

## 📁 Data Structure

Your data directory should have the following structure:

```
subtomos/
├── fitting_subtomos/
│   └── subtomo0/
│       ├── subtomo_000.pt
│       ├── subtomo_001.pt
│       └── ...
└── val_subtomos/
    └── subtomo0/
        ├── subtomo_000.pt
        ├── subtomo_001.pt
        └── ...
```

Note: Unlike Noise2Noise, N2V only requires a single set of subtomograms (subtomo0).

## 🚀 GPU Acceleration

### CUDA (NVIDIA GPUs)

1. **Install CUDA Toolkit** from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. **Install NVIDIA drivers**
3. **Run CUDA installation script**:
   ```bash
   ./install_cuda_torch.sh
   ```
4. **Verify CUDA**:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### MPS (Apple Silicon)

1. **Ensure macOS 12.3+** (Monterey) or later
2. **Use Apple Silicon** (M1, M2, etc.)
3. **PyTorch will automatically use MPS** when available
4. **Verify MPS**:
   ```bash
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```

### CPU Fallback

If no GPU acceleration is available, the environment will automatically fall back to CPU.

## 📊 Training Examples

### Basic Training

```python
from src.ddw.fit_n2v_model import fit_n2v_model

fit_n2v_model(
    unet_params_dict={'chans': 64, 'num_downsample_layers': 3, 'drop_prob': 0.3},
    adam_params_dict={'lr': 4e-2},
    num_epochs=400,
    batch_size=32,
    num_workers=8,
    gpu=0,
    subtomo_size=128,
    mw_angle=50,
    subtomo_dir="path/to/subtomos",
    project_dir="path/to/project",
    n2v_masked_pixel_percentage=0.2,
    n2v_roi_size=11,
    n2v_strategy="uniform",
)
```

### Multi-GPU Training

```bash
python src/ddw/fit_n2v_model.py \
    --gpu [0,1,2] \
    --batch-size 64 \
    --num-workers 16 \
    # ... other parameters
```

### MPS Training (macOS)

```bash
python src/ddw/fit_n2v_model.py \
    --accelerator mps \
    --batch-size 16 \
    # ... other parameters
```

## 🔬 Performance Optimization

### CUDA Optimization
- Use `pin_memory=True` in DataLoader
- Use `persistent_workers=True`
- Use mixed precision (`precision=16`)
- Use multiple GPUs

### MPS Optimization
- Use smaller batch sizes (MPS has memory limitations)
- Monitor memory usage in Activity Monitor

### General Optimization
- Set `num_workers` to your CPU core count
- Use `shuffle=True` for better training
- Use appropriate `batch_size` based on GPU memory
- Use `n2v_strategy="uniform"` for faster training

## 📚 Documentation

- [N2V Implementation Details](N2V_IMPLEMENTATION.md) - Detailed explanation of the N2V implementation
- [Environment Guide](ENVIRONMENT_GUIDE.md) - Complete guide to setting up the environment
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)

## 🐛 Troubleshooting

### Common Issues

**CUDA not detected:**
1. Verify CUDA toolkit is installed: `nvcc --version`
2. Check NVIDIA drivers: `nvidia-smi`
3. Reinstall PyTorch with CUDA: `./install_cuda_torch.sh`

**MPS not available on macOS:**
1. Ensure macOS 12.3+ (Monterey) or later
2. Ensure you're using Apple Silicon (M1, M2, etc.)
3. Update to the latest PyTorch version

**Import errors:**
1. Ensure virtual environment is activated: `source .venv/bin/activate`
2. Check that all dependencies are installed: `uv pip install -r requirements.txt`

**Out of memory errors:**
1. Reduce `batch_size`
2. Reduce `subtomo_size`
3. Use smaller `n2v_masked_pixel_percentage`
4. Use `precision=16` for mixed precision training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Noise2Void Paper](https://arxiv.org/abs/1811.10980) - Original N2V paper
- [CAREamics Library](https://github.com/CAREamics/careamics) - Reference implementation
- [PyTorch Lightning](https://lightning.ai/) - Training framework
- [uv](https://github.com/astral-sh/uv) - Python package manager

## 📞 Support

For questions, issues, or feature requests, please open an issue on the GitHub repository.

---

**Note**: This project is designed to work with both CUDA (NVIDIA GPUs) and MPS (Apple Silicon) backends. The environment setup script will automatically detect your hardware and install the appropriate PyTorch version.
