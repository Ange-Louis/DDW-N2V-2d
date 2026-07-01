# %%
"""
Prepare data for Noise2Void training.

This script extracts subtomograms for N2V training, which only requires a single set
of tomograms (unlike Noise2Noise which requires pairs).
"""

import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional

import torch
import typer
from typer_config import conf_callback_factory
from typing_extensions import Annotated

from src.ddw.utils.load_function_args_from_yaml_config import (
    load_function_args_from_yaml_config,
)
from src.ddw.utils.mrctools2 import load_data, collect_data
from src.ddw.utils.subtomos2 import extract_subtomos

loader = lambda yaml_config_file: load_function_args_from_yaml_config(
    function=prepare_n2v_data, yaml_config_file=yaml_config_file
)
callback = conf_callback_factory(loader)


def prepare_n2v_data(
    tomo_files: Annotated[
        List[Path],
        typer.Option(
            help="List of paths to tomograms (mrc files). Unlike N2N, N2V only requires a single set of tomograms."
        ),
    ],
    subtomo_size: Annotated[
        int,
        typer.Option(
            help="Size of the square subtomograms to extract for model fitting. This value must be divisible by 2^{num_downsample_layers}, where {num_downsample_layers} is the number of downsampling layers used in the U-Net."
        ),
    ],
    val_fraction: Annotated[
        float,
        typer.Option(
            help="Fraction of subtomograms to use for validation. Increasing this fraction will decrease the number of subtomograms used for model fitting."
        ),
    ] = 0.1,
    mask_files: Annotated[
        List[Path],
        typer.Option(
            help="The masks aren't working at the moment\\n                \\nList of paths to binary masks (mrc files) that outline the region of interest in the tomograms to guide subtomogram extraction. The DeepDeWedge reconstruction of areas outside the mask may be less accurate. If no mask_files are provided, the entire tomogram is used for subtomogram extraction."
        ),
    ] = [],
    min_nonzero_mask_fraction_in_subtomo: Annotated[
        Optional[float],
        typer.Option(
            help="The masks aren't working at the moment\\n                \\nMinimum fraction of pixels in a subtomogram that correspond to nonzero pixels in the mask. If mask_files are provided, this parameter has to be provided as well. If no mask_files are provided, this parameter is ignored."
        ),
    ] = 0.3,
    subtomo_extraction_strides: Annotated[
        Optional[List[int]],
        typer.Option(
            help="List of 2 integers specifying the 2D Strides used for subtomogram extraction. If set to None, stride 'subtomo_size' is used in all 2 directions. Smaller strides result in more sub-tomograms being extracted."
        ),
    ] = None,
    pad_before_subtomo_extraction: Annotated[
        bool,
        typer.Option(
            help="Whether to pad the tomograms before extracting subtomograms."
        ),
    ] = False,
    extract_larger_subtomos_for_rotating: Annotated[
        bool,
        typer.Option(
            help="If True, larger subtomograms with a size of 'subtomo_size*sqrt(2)' will be extracted in order to avoid boundary effects when rotating the subtomograms."
        ),
    ] = False,
    standardize_full_tomos: Annotated[
        bool,
        typer.Option(
            help="If 'True', the tomograms will be standardized (mean=0, std=1) before extracting the subtomograms. This is useful for tomograms with low pixel intensities, and DDW can fail when processing such tomograms without standardization."
        ),
    ] = False,
    subtomo_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Where to save the subtomograms. If not provided, the subtomograms will be saved to '{project_dir}/subtomos'."
        ),
    ] = None,
    data_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Where to save the initial tomograms. If not provided, the tomograms will be saved to '{project_dir}/tomos'."
        ),
    ] = None,
    project_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="If 'subtomo_dir' is not provided, the subtomogram directory will saved to '{project_dir}/subtomos'."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Whether to overwrite the existing subtomo_dir if it already exists. If False, the function will raise an error if the directory already exists."
        ),
    ] = False,
    seed: Annotated[
        Optional[int],
        typer.Option(help="Controls the randomness of the validation data selection."),
    ] = None,
    verbose: Annotated[bool, typer.Option()] = False,
    config: Annotated[
        Optional[Path],
        typer.Option(
            callback=callback,
            is_eager=True,
            help="Path to a yaml file containing the arguments for this function. Command line arguments will overwrite the ones in the yaml file.",
        ),
    ] = None,
):
    """
    Extract square sub-tomograms for Noise2Void training.
    
    Unlike prepare_data (for N2N), this function only requires a single set of tomograms
    since N2V is a self-supervised method that doesn't need pairs.
    
    Typically the first command to run for N2V training.
    """
    # check if mask_files are provided properly
    if len(mask_files) == 0:
        mask_files = [None] * len(tomo_files)
        min_nonzero_mask_fraction_in_subtomo = 0.0
    else:
        if min_nonzero_mask_fraction_in_subtomo is None:
            raise ValueError(
                "min_nonzero_mask_fraction_in_subtomo must be provided if mask_files are provided"
            )
        
    if verbose:
        print(f"Starting subtomogram extraction from {len(tomo_files)} tomogram(s) for N2V training.")

    if data_dir is None:
        if project_dir is not None:
            data_dir = f"{project_dir}/tomos"
        else:
            raise ValueError("data_dir must be provided if project_dir is not provided")

    if subtomo_dir is None:
        if project_dir is not None:
            subtomo_dir = f"{project_dir}/subtomos"
        else:
            raise ValueError("subtomo_dir must be provided if project_dir is not provided")

    for d in [data_dir, subtomo_dir]:
        if os.path.exists(d):
            if overwrite:
                if verbose:
                    print(f"Removing existing directory '{d}'.")
                shutil.rmtree(d)
            else:
                raise ValueError(
                    f"Directory '{d}' already exists. Set 'overwrite=True' to remove it."
                )
        os.makedirs(d, exist_ok=True)

    total_fitting_counter, total_val_counter = 0, 0

    for num_tomos, (tomo_file, mask_file) in enumerate(zip(tomo_files, mask_files)):
        fitting_counter, val_counter = 0, 0
        
        tomo_name = Path(tomo_file).stem
        
        # create output directories specifically for these tomograms
        # For N2V, we only need subtomo0 (no subtomo1 needed)
        tomo_dir, fitting_subtomo_dir, val_subtomo_dir = setup_n2v_tomo_dir(
            data_dir=data_dir,
            subtomo_dir=subtomo_dir,
            tomo_name=tomo_name
        )

        if mask_file is not None:
            mask_name = Path(mask_file).stem
            mask_dir = Path(data_dir) / "masks" / mask_name
            os.makedirs(mask_dir, exist_ok=True)

        # Collect data from tomogram files
        if verbose:
            print(f"Processing tomogram: '{tomo_name}'")

        collect_data(image_file=tomo_file, output_dir=tomo_dir)

        if mask_file is not None:
            collect_data(image_file=mask_file, output_dir=mask_dir)
        
        # actual subtomogram extraction
        tomo_tensorfiles = sorted(Path(tomo_dir).glob("*.pt"), key=lambda x: int(x.stem))

        for tomo_tensorfile in tomo_tensorfiles:
            tomo = load_data(tomo_tensorfile).float()
            
            if standardize_full_tomos:
                print(
                    f"Standardizing tomogram '{Path(tomo_tensorfile).stem}' before extracting sub-tomograms."
                )
                tomo -= tomo.mean()
                tomo /= tomo.std()
            else:
                std = tomo.std()
                if std < 1e-3:
                    print(f"\\n                        WARNING: Standard deviation of '{Path(tomo_tensorfile).stem}' is low ({std}), which may lead to issues during model fitting!\\n                        \\nConsider setting 'standardize_full_tomos=True'.\\n                        \\nIf you do so, you must also set 'standardize_full_tomos=True' for 'ddw refine-tomogram'.\
                ")
          
            subtomos, start_coords = extract_subtomos(
                tomo=tomo,
                subtomo_size=subtomo_size,
                subtomo_extraction_strides=subtomo_extraction_strides,
                enlarge_subtomos_for_rotating=extract_larger_subtomos_for_rotating,
                pad_before_subtomo_extraction=pad_before_subtomo_extraction,
            )

            for idx in range(len(subtomos)):
                torch.save(
                    subtomos[idx].clone(), f"{fitting_subtomo_dir}/{fitting_counter}.pt"
                )
                fitting_counter += 1


        fitting_subtomo_tensorfiles = sorted(Path(fitting_subtomo_dir).glob("*.pt"), key=lambda x: int(x.stem))

        num_val_subtomos = math.ceil(len(fitting_subtomo_tensorfiles) * val_fraction)
        val_ids = (
            random.Random(seed).sample(range(len(fitting_subtomo_tensorfiles)), num_val_subtomos)
            if num_val_subtomos > 0
            else []
        )
        for idx in sorted(val_ids):
            shutil.move(f"{fitting_subtomo_dir}/{idx}.pt", f"{val_subtomo_dir}/{idx}.pt")
            val_counter += 1

        fitting_counter -= val_counter

        total_fitting_counter += fitting_counter
        total_val_counter += val_counter

        if verbose:
            print(f"Done with {num_tomos+1}th sub-tomogram extraction.")
            print(
                f"Saved {fitting_counter} sub-tomograms for model fitting from {num_tomos} tomogram."
            )
            print(
                f"Saved {val_counter} sub-tomograms for validation from {num_tomos} tomogram."
            )

    if verbose:
        print(f"Done with all sub-tomogram extraction.")
        print(
            f"Saved a total of {total_fitting_counter} sub-tomograms for model fitting."
        )
        print(
            f"Saved a total of {total_val_counter} sub-tomograms for validation."
        )

def setup_n2v_tomo_dir(data_dir, subtomo_dir, tomo_name):
    """
    Sets up sub-directories specific to the current tomogram for N2V.
    Unlike N2N, N2V only needs a single set of subtomograms (subtomo0).
    """
    tomo_dir = f"{data_dir}/tomo/{tomo_name}"
    
    # For N2V, we only have subtomo0 (no subtomo1 needed)
    fitting_subtomo_dir = f"{subtomo_dir}/fitting_subtomos/subtomo0/{tomo_name}"
    val_subtomo_dir = f"{subtomo_dir}/val_subtomos/subtomo0/{tomo_name}"

    os.makedirs(tomo_dir, exist_ok=True)
    os.makedirs(fitting_subtomo_dir, exist_ok=True)
    os.makedirs(val_subtomo_dir, exist_ok=True)
    
    return tomo_dir, fitting_subtomo_dir, val_subtomo_dir


# if __name__ == "__main__":
#     prepare_n2v_data(
#         tomo_files=["/path/to/your/tomogram.rec"],
#         subtomo_size=128,
#         project_dir="testing_n2v",
#         overwrite=True,
#         subtomo_extraction_strides=[80, 80],
#         extract_larger_subtomos_for_rotating=False,
#     )
