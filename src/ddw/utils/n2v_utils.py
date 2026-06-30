"""Noise2Void utilities for DDW-N2V-2d."""

import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PixelManipulationStrategy(Enum):
    """Strategy for pixel manipulation in N2V."""
    UNIFORM = "uniform"
    MEDIAN = "median"


@dataclass
class N2VConfig:
    """Configuration for Noise2Void manipulation."""
    masked_pixel_percentage: float = 0.2
    roi_size: int = 11
    strategy: PixelManipulationStrategy = PixelManipulationStrategy.UNIFORM
    remove_center: bool = True
    seed: int = 888


class N2VManipulate:
    """
    Noise2Void pixel manipulation for 2D subtomograms.
    
    This class implements the core N2V manipulation logic:
    - Selects random pixels to mask
    - Replaces them with values from neighboring pixels
    - Returns masked image, original image, and mask
    """
    
    def __init__(self, config: N2VConfig):
        """
        Initialize N2V manipulation.
        
        Parameters
        ----------
        config : N2VConfig
            Configuration for N2V manipulation.
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        
    def _get_stratified_coords(self, shape: tuple, mask_pixel_perc: float) -> np.ndarray:
        """
        Generate coordinates of pixels to mask using stratified sampling.
        
        Parameters
        ----------
        shape : tuple
            Shape of the input (batch, height, width).
        mask_pixel_perc : float
            Percentage of pixels to mask.
            
        Returns
        -------
        np.ndarray
            Array of coordinates of pixels to mask.
        """
        batch_size, height, width = shape
        n_dims = 2  # spatial dimensions
        expected_area_per_pixel = 1 / (mask_pixel_perc / 100)
        grid_size = expected_area_per_pixel ** (1 / n_dims)
        
        grid_height = int(np.ceil(height / grid_size))
        grid_width = int(np.ceil(width / grid_size))
        
        # Create grid coordinates
        coords = []
        for b in range(batch_size):
            for i in range(grid_height):
                for j in range(grid_width):
                    y = i * grid_size + self.rng.uniform(0, grid_size)
                    x = j * grid_size + self.rng.uniform(0, grid_size)
                    y = min(int(y), height - 1)
                    x = min(int(x), width - 1)
                    coords.append([b, y, x])
        
        return np.array(coords)
    
    def _uniform_manipulate(self, patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply uniform pixel manipulation.
        
        Parameters
        ----------
        patch : torch.Tensor
            Input tensor of shape (batch, height, width).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Masked patch and mask.
        """
        device = patch.device
        batch_size, height, width = patch.shape
        
        # Get coordinates of pixels to mask
        subpatch_centers = self._get_stratified_coords(
            (batch_size, height, width), 
            self.config.masked_pixel_percentage
        )
        subpatch_centers = torch.from_numpy(subpatch_centers).to(device)
        
        # Create ROI around each center
        half_size = self.config.roi_size // 2
        roi_span = torch.arange(-half_size, half_size + 1, device=device)
        
        # Remove center if needed
        if self.config.remove_center:
            roi_span = roi_span[roi_span != 0]
        
        # Generate random offsets
        random_increment = roi_span[
            self.rng.integers(
                low=min(roi_span.cpu().numpy()),
                high=max(roi_span.cpu().numpy()) + 1,
                size=(len(subpatch_centers), 2)  # 2 spatial dimensions
            )
        ]
        random_increment = torch.tensor(random_increment, device=device)
        
        # Compute replacement coordinates
        replacement_coords = subpatch_centers.clone()
        replacement_coords[:, 1:] = torch.clamp(
            replacement_coords[:, 1:] + random_increment,
            torch.zeros(2, device=device),
            torch.tensor([height - 1, width - 1], device=device)
        )
        
        # Replace pixels
        transformed_patch = patch.clone()
        replacement_pixels = patch[tuple(replacement_coords.T)]
        transformed_patch[tuple(subpatch_centers.T)] = replacement_pixels
        
        # Create mask
        mask = (transformed_patch != patch).to(dtype=torch.uint8)
        
        return transformed_patch, mask
    
    def _median_manipulate(self, patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply median pixel manipulation (N2V2 style).
        
        Parameters
        ----------
        patch : torch.Tensor
            Input tensor of shape (batch, height, width).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Masked patch and mask.
        """
        device = patch.device
        batch_size, height, width = patch.shape
        
        # Get coordinates of pixels to mask
        subpatch_centers = self._get_stratified_coords(
            (batch_size, height, width), 
            self.config.masked_pixel_percentage
        )
        subpatch_centers = torch.from_numpy(subpatch_centers).to(device)
        
        # Create subpatches around each center
        half_size = self.config.roi_size // 2
        subpatch_size = self.config.roi_size
        
        # Create subpatch coordinates
        offsets = torch.meshgrid(
            torch.arange(-half_size, half_size + 1, device=device),
            torch.arange(-half_size, half_size + 1, device=device),
            indexing='ij'
        )
        offsets = torch.stack(offsets, dim=0)  # (2, roi_size, roi_size)
        
        # Add batch dimension
        offsets = torch.cat([
            torch.zeros(1, subpatch_size, subpatch_size, device=device),
            offsets
        ], dim=0)  # (3, roi_size, roi_size)
        
        # Get subpatches
        subpatches = []
        for center in subpatch_centers:
            # Create coordinates for this subpatch
            center_coords = center.view(3, 1, 1)  # (3, 1, 1)
            subpatch_coords = center_coords + offsets  # (3, roi_size, roi_size)
            subpatch_coords = subpatch_coords.clamp(
                min=0,
                max=torch.tensor([batch_size - 1, height - 1, width - 1], device=device).view(3, 1, 1)
            )
            subpatch = patch[tuple(subpatch_coords)]  # (roi_size, roi_size)
            subpatches.append(subpatch)
        
        subpatches = torch.stack(subpatches, dim=0)  # (n_centers, roi_size, roi_size)
        
        # Create mask to exclude center pixel
        center_idx = half_size
        subpatch_mask = torch.ones(subpatch_size, subpatch_size, dtype=torch.bool, device=device)
        subpatch_mask[center_idx, center_idx] = False
        
        # Apply mask and compute medians
        subpatches_masked = subpatches[:, subpatch_mask]
        medians = subpatches_masked.median(dim=1).values
        
        # Update output
        output_patch = patch.clone()
        output_patch[tuple(subpatch_centers.T)] = medians
        mask = (patch != output_patch).to(dtype=torch.uint8)
        
        return output_patch, mask
    
    def __call__(self, patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply N2V manipulation to a batch of patches.
        
        Parameters
        ----------
        patch : torch.Tensor
            Input tensor of shape (batch, height, width).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Masked patch, original patch, and mask.
        """
        if self.config.strategy == PixelManipulationStrategy.UNIFORM:
            masked_patch, mask = self._uniform_manipulate(patch)
        elif self.config.strategy == PixelManipulationStrategy.MEDIAN:
            masked_patch, mask = self._median_manipulate(patch)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        
        return masked_patch, patch, mask


def n2v_loss(
    prediction: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute Noise2Void loss.
    
    Parameters
    ----------
    prediction : torch.Tensor
        Model prediction.
    original : torch.Tensor
        Original image.
    mask : torch.Tensor
        Mask indicating which pixels were manipulated.
        
    Returns
    -------
    torch.Tensor
        Loss value.
    """
    errors = (original - prediction) ** 2
    loss = torch.sum(errors * mask) / (torch.sum(mask) + 1e-8)
    return loss
