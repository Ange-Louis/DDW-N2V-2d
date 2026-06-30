#!/usr/bin/env python3
"""
Environment check script for DDW-N2V-2d.
This script verifies that all dependencies are installed and checks CUDA/MPS availability.
"""

import sys
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib
import mrcfile
import scipy
import skimage
from typing import Optional


def get_version(module) -> str:
    """Safely get version from a module."""
    try:
        if hasattr(module, '__version__'):
            return module.__version__
        elif hasattr(module, 'version'):
            return module.version
        elif hasattr(module, 'VERSION'):
            return module.VERSION
        else:
            return "unknown"
    except Exception:
        return "unknown"


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def check_python_version() -> None:
    """Check Python version."""
    print_section("Python Version")
    print(f"Python: {sys.version}")
    print(f"Python version info: {sys.version_info}")
    
    if sys.version_info >= (3, 10):
        print("✓ Python version is compatible (>= 3.10)")
    else:
        print("✗ Python version is too old (< 3.10)")


def check_pytorch() -> None:
    """Check PyTorch installation and GPU support."""
    print_section("PyTorch Configuration")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        print("✓ CUDA is available")
    else:
        print("✗ CUDA is not available")
    
    # Check MPS (Apple Metal)
    try:
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        print(f"MPS available: {mps_available}")
        print(f"MPS built: {mps_built}")
        
        if mps_available:
            print("✓ MPS (Apple Metal) is available")
        else:
            print("✗ MPS is not available")
    except Exception as e:
        print(f"MPS check failed: {e}")
    
    # Check current device
    if torch.cuda.is_available():
        print(f"Current device: cuda:{torch.cuda.current_device()}")
    elif mps_available:
        print("Current device: mps")
    else:
        print("Current device: cpu")


def check_dependencies() -> None:
    """Check all required dependencies."""
    print_section("Dependencies Check")
    
    dependencies = {
        "PyTorch Lightning": get_version(pl),
        "NumPy": get_version(np),
        "Matplotlib": get_version(matplotlib),
        "MRCFile": get_version(mrcfile),
        "SciPy": get_version(scipy),
        "scikit-image": get_version(skimage),
    }
    
    # Special handling for starfile
    try:
        import starfile
        dependencies["StarFile"] = get_version(starfile)
    except Exception as e:
        dependencies["StarFile"] = f"import error: {e}"
    
    all_ok = True
    for name, version in dependencies.items():
        print(f"{name}: {version}")
        if version == "unknown" or version.startswith("import error"):
            print(f"  ✗ {name} version not found or import error")
            all_ok = False
        else:
            print(f"  ✓ {name} is installed")
    
    if all_ok:
        print("✓ All dependencies are installed")
    else:
        print("✗ Some dependencies are missing")


def check_project_structure() -> None:
    """Check the project structure."""
    print_section("Project Structure")
    
    import os
    from pathlib import Path
    
    project_root = Path(__file__).parent
    
    expected_files = [
        "pyproject.toml",
        "requirements.txt",
        "N2V_IMPLEMENTATION.md",
        "src/ddw/fit_n2v_model.py",
        "src/ddw/utils/n2v_utils.py",
        "src/ddw/utils/n2v_subtomo_dataset.py",
        "src/ddw/utils/n2v_unet2.py",
    ]
    
    all_exist = True
    for file_path in expected_files:
        full_path = project_root / file_path
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("✓ All expected files are present")
    else:
        print("✗ Some expected files are missing")


def test_n2v_imports() -> None:
    """Test importing N2V modules."""
    print_section("N2V Module Imports")
    
    try:
        from src.ddw.utils.n2v_utils import N2VConfig, N2VManipulate, PixelManipulationStrategy
        print("✓ n2v_utils imported successfully")
    except Exception as e:
        print(f"✗ Failed to import n2v_utils: {e}")
    
    try:
        from src.ddw.utils.n2v_subtomo_dataset import N2VSubtomoDataset
        print("✓ n2v_subtomo_dataset imported successfully")
    except Exception as e:
        print(f"✗ Failed to import n2v_subtomo_dataset: {e}")
    
    try:
        from src.ddw.utils.n2v_unet2 import LitN2VUnet2D
        print("✓ n2v_unet2 imported successfully")
    except Exception as e:
        print(f"✗ Failed to import n2v_unet2: {e}")
    
    try:
        from src.ddw.fit_n2v_model import fit_n2v_model
        print("✓ fit_n2v_model imported successfully")
    except Exception as e:
        print(f"✗ Failed to import fit_n2v_model: {e}")


def test_n2v_functionality() -> None:
    """Test basic N2V functionality."""
    print_section("N2V Functionality Test")
    
    try:
        import torch
        from src.ddw.utils.n2v_utils import N2VConfig, N2VManipulate, PixelManipulationStrategy
        
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
        
        print(f"Input shape: {x.shape}")
        print(f"Masked shape: {masked.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Percentage masked: {mask.float().mean().item() * 100:.2f}%")
        print("✓ N2V manipulation works correctly")
        
    except Exception as e:
        print(f"✗ N2V functionality test failed: {e}")
        import traceback
        traceback.print_exc()


def get_device_recommendation() -> str:
    """Get device recommendation based on available hardware."""
    print_section("Device Recommendation")
    
    if torch.cuda.is_available():
        print("✓ CUDA is available - Use 'gpu=0' or 'gpu=[0,1,...]' in your training scripts")
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✓ MPS is available - Use 'accelerator='mps'' in PyTorch Lightning")
        return "mps"
    else:
        print("✗ No GPU acceleration available - Using CPU")
        return "cpu"


def main() -> None:
    """Run all environment checks."""
    print("\n" + "=" * 60)
    print(" DDW-N2V-2d Environment Check")
    print("=" * 60)
    
    check_python_version()
    check_pytorch()
    check_dependencies()
    check_project_structure()
    test_n2v_imports()
    test_n2v_functionality()
    device = get_device_recommendation()
    
    print("\n" + "=" * 60)
    print(" Environment Check Complete!")
    print("=" * 60)
    print(f"\nRecommended device: {device}")
    print("\nNext steps:")
    print("1. Activate the environment: source .venv/bin/activate")
    print("2. Run your training script with the appropriate device")
    print("3. For CUDA: Use 'gpu=0' parameter")
    print("4. For MPS: Set 'accelerator='mps'' in PyTorch Lightning config")
    print("5. For CPU: No special configuration needed")
    print()


if __name__ == "__main__":
    main()
