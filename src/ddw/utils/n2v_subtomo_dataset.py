"""Noise2Void subtomogram dataset for DDW-N2V-2d."""

import os
from pathlib import Path
import time
import torch
import numpy as np
from torch.utils.data import Dataset

from src.ddw.utils.fourier2 import apply_fourier_mask_to_tomo
from src.ddw.utils.missing_wedge2 import get_missing_wedge_mask, get_rotated_missing_wedge_mask
from src.ddw.utils.rotation2 import rotate_area
from src.ddw.utils.n2v_utils import N2VManipulate, N2VConfig, PixelManipulationStrategy

BASE_SEED = 888


def safe_load(file_path, max_retries=3, delay=1):
    """Safely load a torch file with retries."""
    for attempt in range(max_retries):
        try:
            return torch.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}")
            if attempt == max_retries - 1:
                raise e
            print(f"Error message is: {e}")
            print(f"Retrying in {delay} seconds")
            time.sleep(delay)


class N2VSubtomoDataset(Dataset):
    """
    A torch dataset for Noise2Void training on subtomograms.
    
    Unlike the standard SubtomoDataset which requires pairs of subtomograms
    (subtomo0 and subtomo1), this dataset only needs a single set of subtomograms
    and applies N2V manipulation to create training pairs.
    
    The directory 'subtomo_dir' should contain subtomograms in .pt format.
    """

    def __init__(
        self,
        subtomo_dir: str,
        crop_subtomos_to_size: int,
        mw_angle: float,
        rotate_subtomos: bool = True,
        deterministic_rotations: bool = False,
        n2v_config: Optional[N2VConfig] = None,
    ):
        """
        Initialize N2V subtomogram dataset.
        
        Parameters
        ----------
        subtomo_dir : str
            Path to directory containing subtomograms.
        crop_subtomos_to_size : int
            Size to crop subtomograms to.
        mw_angle : float
            Missing wedge angle in degrees.
        rotate_subtomos : bool
            Whether to rotate subtomograms during training.
        deterministic_rotations : bool
            Whether to use deterministic rotations.
        n2v_config : N2VConfig, optional
            Configuration for N2V manipulation. If None, default config is used.
        """
        super().__init__()
        self.subtomo_dir = subtomo_dir
        self.crop_subtomos_to_size = crop_subtomos_to_size
        self.mw_angle = mw_angle
        self.rotate_subtomos = rotate_subtomos
        self.deterministic_rotations = deterministic_rotations
        
        # Initialize N2V manipulation
        if n2v_config is None:
            self.n2v_config = N2VConfig(
                masked_pixel_percentage=0.2,
                roi_size=11,
                strategy=PixelManipulationStrategy.UNIFORM,
                remove_center=True,
                seed=BASE_SEED
            )
        else:
            self.n2v_config = n2v_config
        
        self.n2v_manipulate = N2VManipulate(self.n2v_config)
        
        # Load subtomogram files
        self.subtomo_path = Path(self.subtomo_dir)
        self.subtomo_files = sorted(list(self.subtomo_path.rglob("*.pt")))

    @property
    def rotate_subtomos(self):
        return self._rotate_subtomos

    @rotate_subtomos.setter
    def rotate_subtomos(self, rotate_subtomos):
        if not isinstance(rotate_subtomos, bool):
            raise ValueError("rotate_subtomos must be a boolean")
        self._rotate_subtomos = rotate_subtomos

    def _sample_rot_angle(self, index):
        seed = BASE_SEED + index if self.deterministic_rotations else None
        rng = np.random.default_rng(seed)
        rot_angle = torch.tensor(rng.uniform(0, 360), dtype=torch.float32)
        return rot_angle

    def __len__(self):
        return len(self.subtomo_files)

    def __getitem__(self, index):
        # Load subtomogram
        subtomo_file = str(self.subtomo_files[index])
        subtomo = safe_load(subtomo_file)
        
        # Rotate subtomogram
        if self.rotate_subtomos:
            rot_angle = self._sample_rot_angle(index)
            subtomo = rotate_area(
                subtomo,
                rot_angle=rot_angle,
                output_shape=2 * [self.crop_subtomos_to_size],
            )
            
            # Add missing wedge
            mw_mask = get_missing_wedge_mask(
                grid_size=2 * [self.crop_subtomos_to_size],
                mw_angle=self.mw_angle,
                device=subtomo.device,
            )
            rot_mw_mask = get_rotated_missing_wedge_mask(
                grid_size=2 * [self.crop_subtomos_to_size],
                mw_angle=self.mw_angle,
                rot_angle=rot_angle,
                device=subtomo.device,
            )
        else:
            mw_mask = get_missing_wedge_mask(
                grid_size=subtomo.shape,
                mw_angle=self.mw_angle,
                device=subtomo.device,
            )
            rot_mw_mask = mw_mask
            rot_angle = 0

        # Apply missing wedge to create model input
        model_input = apply_fourier_mask_to_tomo(subtomo, mw_mask)
        
        # Apply N2V manipulation to create training pair
        # We need to add batch dimension for N2V manipulation
        model_input_batch = model_input.unsqueeze(0)  # (1, H, W)
        
        masked_input, original, mask = self.n2v_manipulate(model_input_batch)
        
        # Remove batch dimension
        masked_input = masked_input.squeeze(0)
        original = original.squeeze(0)
        mask = mask.squeeze(0)
        
        # The model target is the original (unmasked) version
        # But we need to apply the same missing wedge mask to it
        model_target = apply_fourier_mask_to_tomo(original, mw_mask)
        
        item = {
            "model_input": masked_input,
            "model_target": model_target,
            "mw_mask": mw_mask,
            "rot_mw_mask": rot_mw_mask,
            "subtomo_file": subtomo_file,
            "rot_angle": rot_angle,
            "n2v_mask": mask,  # The N2V mask indicating which pixels were manipulated
        }
        return item
