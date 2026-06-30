# Noise2Void Implementation for DDW-N2V-2d

This document describes the Noise2Void (N2V) implementation that has been added to the DDW-N2V-2d project, replacing the original Noise2Noise (N2N) approach.

## Overview

### Key Differences: N2N vs N2V

| Aspect | Noise2Noise (Original) | Noise2Void (New) |
|--------|------------------------|------------------|
| **Data Requirements** | Requires pairs of subtomograms (subtomo0 and subtomo1) | Only needs a single set of subtomograms |
| **Supervision** | Supervised learning (uses subtomo1 as target) | Self-supervised learning (creates artificial targets) |
| **Training Data** | Two noisy versions of the same scene | Single noisy version with masked pixels |
| **Target** | The other noisy acquisition | Original image with masked pixels replaced |
| **Loss Function** | MSE between prediction and subtomo1 | MSE between prediction and original, only on masked pixels |

## Implementation Details

### Core Components

1. **`n2v_utils.py`**: Contains the core N2V manipulation logic
   - `N2VConfig`: Configuration dataclass for N2V parameters
   - `PixelManipulationStrategy`: Enum for manipulation strategies (UNIFORM, MEDIAN)
   - `N2VManipulate`: Main class that handles pixel masking and replacement
   - `n2v_loss()`: Loss function that only considers masked pixels

2. **`n2v_subtomo_dataset.py`**: Dataset class for N2V training
   - `N2VSubtomoDataset`: Replaces the original `SubtomoDataset`
   - Only requires a single directory of subtomograms
   - Applies N2V manipulation to create training pairs
   - Maintains compatibility with missing wedge handling

3. **`n2v_unet2.py`**: U-Net model adapted for N2V
   - `LitN2VUnet2D`: PyTorch Lightning module for N2V training
   - Uses the N2V loss function
   - Maintains all the original functionality (missing wedge updates, normalization, etc.)

4. **`fit_n2v_model.py`**: Main training script
   - Similar interface to the original `fit_model2.py`
   - Additional N2V-specific parameters

### N2V Strategies

The implementation supports two main strategies from the CAREamics library:

1. **Uniform Strategy (Original N2V)**
   - Replaces masked pixels with values from neighboring pixels
   - Uses a uniform random selection from a local region
   - Configured with `n2v_strategy="uniform"`

2. **Median Strategy (N2V2)**
   - Replaces masked pixels with the median of their neighborhood
   - More robust to outliers
   - Configured with `n2v_strategy="median"`

## Usage

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
    # N2V specific parameters
    n2v_masked_pixel_percentage=0.2,  # Percentage of pixels to mask
    n2v_roi_size=11,                   # Size of the region for pixel replacement
    n2v_strategy="uniform",           # or "median" for N2V2
)
```

### Configuration Parameters

#### N2V-Specific Parameters

- `n2v_masked_pixel_percentage` (float, default: 0.2)
  - Percentage of pixels to mask in each subtomogram
  - Typical range: 0.05 to 10.0

- `n2v_roi_size` (int, default: 11)
  - Size of the region of interest for pixel manipulation
  - Must be an odd number
  - Typical range: 3 to 21

- `n2v_strategy` (str, default: "uniform")
  - Strategy for pixel manipulation
  - Options: "uniform" (original N2V) or "median" (N2V2)

### Data Preparation

Unlike N2N, N2V only requires a single set of subtomograms. Your data directory should have the following structure:

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

Note: You no longer need the `subtomo1` directory that was required for N2N.

## Migration from N2N to N2V

### Changes Required

1. **Data Structure**
   - Remove the `subtomo1` directory from your data
   - Keep only `subtomo0` (or rename it if you prefer)

2. **Training Script**
   - Replace `from src.ddw.fit_model2 import fit_model2` with `from src.ddw.fit_n2v_model import fit_n2v_model`
   - Update the function call to use the new parameters

3. **Configuration Files**
   - Update any YAML configuration files to include the new N2V parameters
   - Remove parameters specific to N2N (like references to subtomo1)

### Example Migration

**Before (N2N):**
```python
from src.ddw.fit_model2 import fit_model2

fit_model2(
    unet_params_dict={'chans': 64, 'num_downsample_layers': 3, 'drop_prob': 0.3},
    adam_params_dict={'lr': 4e-2},
    num_epochs=400,
    batch_size=32,
    subtomo_size=128,
    mw_angle=50,
    subtomo_dir="testing/subtomos",  # Contains both subtomo0 and subtomo1
    project_dir="testing2",
)
```

**After (N2V):**
```python
from src.ddw.fit_n2v_model import fit_n2v_model

fit_n2v_model(
    unet_params_dict={'chans': 64, 'num_downsample_layers': 3, 'drop_prob': 0.3},
    adam_params_dict={'lr': 4e-2},
    num_epochs=400,
    batch_size=32,
    subtomo_size=128,
    mw_angle=50,
    subtomo_dir="testing/subtomos",  # Only needs subtomo0
    project_dir="testing2",
    n2v_masked_pixel_percentage=0.2,
    n2v_roi_size=11,
    n2v_strategy="uniform",
)
```

## Advanced Features

### StructN2V Support

The implementation can be extended to support StructN2V by adding structural masking patterns. This is available in the CAREamics library and can be added to this implementation if needed.

### Mixed Training

You can potentially combine N2V and N2N approaches by:
1. Using N2V for most of the training
2. Fine-tuning with N2N if you have pairs available

## Performance Considerations

1. **Memory Usage**: N2V requires slightly more memory as it needs to store the original image, masked image, and mask for each batch.

2. **Training Time**: N2V training is typically slightly slower than N2N due to the additional manipulation steps.

3. **Quality**: N2V often produces better results than N2N, especially when only single acquisitions are available.

## References

- Krull, A., Buchholz, T.O., Jug, F., et al. (2019). Noise2Void - Learning Denoising from Single Noisy Images. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
- CAREamics library: https://github.com/CAREamics/careamics

## Files Added/Modified

### New Files
- `src/ddw/utils/n2v_utils.py` - Core N2V utilities
- `src/ddw/utils/n2v_subtomo_dataset.py` - N2V dataset implementation
- `src/ddw/utils/n2v_unet2.py` - N2V U-Net model
- `src/ddw/fit_n2v_model.py` - Main training script

### Modified Files
- None (all changes are additive to maintain backward compatibility)

## Testing

To test the N2V implementation:

```bash
# Run a simple test
python -c "
from src.ddw.utils.n2v_utils import N2VManipulate, N2VConfig, PixelManipulationStrategy
import torch

# Create test data
x = torch.randn(1, 64, 64)

# Create N2V manipulator
config = N2VConfig(
    masked_pixel_percentage=0.2,
    roi_size=11,
    strategy=PixelManipulationStrategy.UNIFORM
)
manipulator = N2VManipulate(config)

# Apply manipulation
masked, original, mask = manipulator(x)
print(f'Input shape: {x.shape}')
print(f'Masked shape: {masked.shape}')
print(f'Mask shape: {mask.shape}')
print(f'Percentage masked: {mask.float().mean().item() * 100:.2f}%')
"
```

## Troubleshooting

1. **CUDA Out of Memory**: Reduce `batch_size` or `subtomo_size`.

2. **Slow Training**: 
   - Reduce `n2v_masked_pixel_percentage` (e.g., from 0.2 to 0.1)
   - Reduce `n2v_roi_size` (e.g., from 11 to 7)
   - Use `n2v_strategy="uniform"` instead of "median" (faster)

3. **Poor Results**:
   - Try increasing `n2v_masked_pixel_percentage` (e.g., to 0.3 or 0.5)
   - Try using `n2v_strategy="median"` (N2V2)
   - Ensure your data is properly normalized

## Compatibility

This implementation maintains backward compatibility with the existing DDW-N2V-2d codebase. You can continue to use the original N2N implementation by using `fit_model2.py` as before.

To switch between N2N and N2V:
- Use `fit_model2.py` for N2N (requires pairs)
- Use `fit_n2v_model.py` for N2V (single set)
