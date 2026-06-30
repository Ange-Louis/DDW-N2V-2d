# %%
"""
Refine tomograms using a Noise2Void model.

This script uses a fitted N2V U-Net to denoise tomograms and fill in their missing wedge.
Unlike the N2N version, this only requires a single set of tomograms.
"""

import math
import os
from pathlib import Path
from typing import List, Optional

import torch
import tqdm
import typer
from torch.utils.data import DataLoader, TensorDataset
from typer_config import conf_callback_factory
from typing_extensions import Annotated

from src.ddw.fit_n2v_model import LitN2VUnet2D
from src.ddw.utils.fourier2 import apply_fourier_mask_to_tomo
from src.ddw.utils.load_function_args_from_yaml_config import \
    load_function_args_from_yaml_config
from src.ddw.utils.missing_wedge2 import get_missing_wedge_mask
from src.ddw.utils.mrctools2 import load_data, save_mrc_data, load_2d_data
from src.ddw.utils.normalization2 import get_avg_model_input_mean_and_std
from src.ddw.utils.subtomos2 import extract_subtomos, reassemble_subtomos

loader = lambda yaml_config_file: load_function_args_from_yaml_config(
    function=refine_n2v_tomogram, yaml_config_file=yaml_config_file
)
callback = conf_callback_factory(loader)


def refine_n2v_tomogram(
    tomo_files: Annotated[
        List[Path],
        typer.Option(
            help="List of paths to tomograms (mrc files). Unlike N2N, N2V only requires a single set of tomograms."
        ),
    ],
    model_checkpoint_file: Annotated[
        Path,
        typer.Option(
            help="Path to a model checkpoint file (.ckpt extension). Checkpoints saved during N2V model fitting can be found in the 'logdir' directory specified for the 'fit-n2v-model' command."
        ),
    ],
    subtomo_size: Annotated[
        int,
        typer.Option(
            help="Size of the cubic subtomograms to extract. This should be the same as the subtomo_size used during model fitting."
        ),
    ],
    mw_angle: Annotated[
        int, typer.Option(help="Width of the missing wedge in degrees.")
    ],
    subtomo_overlap: Annotated[
        Optional[int],
        typer.Option(
            help="Overlap between subtomograms. This determines the stride of the sliding window used to extract subtomograms. If 'None', this is set to '1/3 * subtomo_size'."
        ),
    ] = None,
    standardize_full_tomos: Annotated[
        bool,
        typer.Option(
            help="Set to 'True' if and only if 'standardize_full_tomos' was 'True' for 'ddw prepare-n2v-data'."
        ),
    ] = False,
    recompute_normalization: Annotated[
        bool,
        typer.Option(
            help="Whether to recompute the mean and variance used to normalize the tomograms (see Appendix B in the paper). If `False`, the mean and variance of model inputs calculated during model fitting will be used. If `True`, the average model input mean and variance will be computed for each tomogram individually. We recommend setting this to `True`. If you apply a model to a tomogram that was not used for model fitting or if the means and variances of the tomograms during model fitting are considerably different, recomputing the normalization is expected to be very beneficial for tomogram refinement."
        ),
    ] = True,
    batch_size: Annotated[
        int, typer.Option(help="Batch size for processing subtomograms.")
    ] = 1,
    return_tomos: Annotated[
        bool,
        typer.Option(
            help="Whether to return the refined tomograms as a list of tensors. If 'False', the refined tomograms will only be saved to 'output_dir', and the function returns nothing."
        ),
    ] = False,
    data_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Where to save the initial tomogram slices. If not provided, the tomograms will be saved to '{project_dir}/tomos'."
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Where to save the refined tomograms. If not provided, either 'project_dir' has to be provided or 'return_subtomos' must be 'True'."
        ),
    ] = None,
    project_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to the project directory. If not provided the refined tomograms will be saved to {project_dir}/refined_tomograms. If 'return_subtomos' is False, and 'output_dir' is not provided, this has to be provided."
        ),
    ] = None,
    num_workers: Annotated[
        int,
        typer.Option(
            help="Number of CPU workers to use during the recomputation of the normalization statistics and dataloading for refining the tomograms."
        ),
    ] = 0,
    gpu: Annotated[
        Optional[List[int]],
        typer.Option(
            help="GPU id on which to run the model. If None, the model will be run on the CPU. Currently, only a single GPU is supported. Providing multiple GPUs will result in a warning and only the first GPU will be used."
        ),
    ] = None,
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
    Use a fitted N2V U-Net to denoise tomograms and to fill in their missing wedge.
    
    Unlike refine_tomogram (for N2N), this function only requires a single set of tomograms
    since N2V is a self-supervised method.
    
    Typically run after `fit-n2v-model`.
    """
    if data_dir is None:
        if project_dir is not None:
            data_dir = f"{project_dir}/tomos"
        else:
            raise ValueError("data_dir must be provided if project_dir is not provided")
    if output_dir is None:
        if project_dir is not None:
            output_dir = f"{project_dir}/refined_tomograms"
        elif project_dir is None and return_tomos is False:
            raise ValueError(
                "If return_tomos is False, output_dir or project_dir must be provided, otherwise the refined tomograms will be lost."
            )
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if return_tomos is False and output_dir is None:
        raise ValueError(
            "If return_tomos is False, output_dir or project_dir must be provided, otherwise the refined tomograms will be lost."
        )
    if return_tomos:
        tomo_ref = []

    if subtomo_overlap is None:
        subtomo_overlap = int(math.ceil(subtomo_size / 3))

    device = "cpu" if gpu is None else f"cuda:{gpu[0]}"
    
    # Load N2V model instead of N2N model
    lightning_model = (
        LitN2VUnet2D.load_from_checkpoint(model_checkpoint_file).to(device).eval()
    )

    with torch.no_grad():
        for k, tomo_file in enumerate(tomo_files):
            tomo_name = Path(tomo_file).stem

            tomo_to_refine = load_2d_data(tomo_file)

            print(f"Refining {k}th tomogram: '{tomo_name}'")

            if len(tomo_to_refine.shape) == 3:
                refined = refine_3d_n2v(
                    tomo_to_refine=tomo_to_refine,
                    tomo_name=tomo_name,
                    tomo_file=tomo_file,
                    data_dir=data_dir,
                    lightning_model=lightning_model,
                    subtomo_size=subtomo_size,
                    subtomo_overlap=subtomo_overlap,
                    mw_angle=mw_angle,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    standardize_full_tomos=standardize_full_tomos,
                    recompute_normalization=recompute_normalization
                )
            else:
                refined = refine_2d_n2v(
                    tomo_name=tomo_name,
                    tomo_file=tomo_file,
                    lightning_model=lightning_model,
                    subtomo_size=subtomo_size,
                    subtomo_overlap=subtomo_overlap,
                    mw_angle=mw_angle,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    standardize_full_tomos=standardize_full_tomos,
                    recompute_normalization=recompute_normalization
                )

            if return_tomos:
                tomo_ref.append(refined)
                
            if output_dir is not None:
                outfile = f"{output_dir}/{tomo_name}_refined.mrc"
                print(f"Saving refined tomogram to {outfile}")
                save_mrc_data(refined.cpu(), f"{outfile}", save=True)
    
    if return_tomos:
        return tomo_ref


def _refine_single_tomogram(
    tomo,
    lightning_model,
    subtomo_size,
    subtomo_overlap,
    mw_angle,
    normalization_loc,
    normalization_scale,
    num_workers=0,
    batch_size=1,
    pbar_desc="Refining tomogram",
):
    """Refine a single tomogram slice."""
    # apply missing wedge mask here to be more consistent with data during model fitting
    mw_mask = get_missing_wedge_mask(tomo.shape, mw_angle, device=tomo.device)
    tomo = apply_fourier_mask_to_tomo(tomo, mw_mask)

    tomo = (tomo / tomo.std()) * torch.tensor(normalization_scale).to(tomo.device)
    tomo = tomo - tomo.mean() + torch.tensor(normalization_loc).to(tomo.device)

    subtomos, subtomo_start_coords = extract_subtomos(
        tomo=tomo.cpu(),
        subtomo_size=subtomo_size,
        subtomo_extraction_strides=2 * [subtomo_size - subtomo_overlap],
        enlarge_subtomos_for_rotating=False,
        pad_before_subtomo_extraction=True,
    )
    
    subtomos = TensorDataset(torch.stack(subtomos))
    subtomo_loader = DataLoader(
        subtomos,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    model_outputs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(subtomo_loader, desc=pbar_desc):
            batch_subtomos = batch[0].to(lightning_model.device)
            model_output = lightning_model(batch_subtomos)
            model_output = model_output.detach().cpu()
            model_outputs.append(model_output)
    
    model_outputs = list(torch.concat(model_outputs, 0))

    tomo_ref = reassemble_subtomos(
        subtomos=model_outputs,
        subtomo_start_coords=subtomo_start_coords,
        subtomo_overlap=subtomo_overlap,
        crop_to_size=tomo.shape,
    )
    return tomo_ref


def refine_3d_n2v(
    tomo_to_refine,
    tomo_name,
    tomo_file,
    data_dir,
    lightning_model,
    subtomo_size,
    subtomo_overlap,
    mw_angle,
    num_workers,
    batch_size,
    standardize_full_tomos,
    recompute_normalization,
):
    """Refine a 3D tomogram using N2V model."""
    if recompute_normalization:
        loc, scale = get_avg_model_input_mean_and_std(
            tomo_file=tomo_file,
            subtomo_size=subtomo_size,
            subtomo_extraction_strides=2 * [subtomo_size - subtomo_overlap],
            mw_angle=mw_angle,
            batch_size=batch_size,
            standardize=standardize_full_tomos,
            num_workers=num_workers,
            verbose=True,
        )
    else:
        loc, scale = (
            lightning_model.unet.normalization_loc.clone().detach().item(),
            lightning_model.unet.normalization_scale.clone().detach().item(),
        )

    # For N2V, we only have a single tomogram directory (not tomo0/tomo1)
    tomo_dir = f"{data_dir}/tomo/{tomo_name}"
    tomo_tensorfiles = sorted(Path(tomo_dir).glob("*.pt"), key=lambda x: int(x.stem))

    refined = torch.empty(tomo_to_refine.shape[0], 0, tomo_to_refine.shape[2])

    for k, tomo_tensorfile in enumerate(tomo_tensorfiles):
        tomo = load_data(tomo_tensorfile).float()

        print(f"Processing slice {k} from '{tomo_name}'")

        t_ref = _refine_single_tomogram(
            tomo=tomo,
            lightning_model=lightning_model,
            subtomo_size=subtomo_size,
            subtomo_overlap=subtomo_overlap,
            mw_angle=mw_angle,
            normalization_loc=loc,
            normalization_scale=scale,
            num_workers=num_workers,
            batch_size=batch_size,
            pbar_desc=f"Slice {k} of {tomo_name}",
        )

        t_ref = t_ref.unsqueeze(1)
        refined = torch.cat([refined, t_ref], dim=1)
    
    return refined


def refine_2d_n2v(
    tomo_name,
    tomo_file,
    lightning_model,
    subtomo_size,
    subtomo_overlap,
    mw_angle,
    num_workers,
    batch_size,
    standardize_full_tomos,
    recompute_normalization
):
    """Refine a 2D tomogram using N2V model."""
    if recompute_normalization:
        loc, scale = get_avg_model_input_mean_and_std(
            tomo_file=tomo_file,
            subtomo_size=subtomo_size,
            subtomo_extraction_strides=2 * [subtomo_size - subtomo_overlap],
            mw_angle=mw_angle,
            batch_size=batch_size,
            standardize=standardize_full_tomos,
            num_workers=num_workers,
            verbose=True
        )
    else:
        loc, scale = (
            lightning_model.unet.normalization_loc.clone().detach().item(),
            lightning_model.unet.normalization_scale.clone().detach().item()
        )

    tomo = load_2d_data(tomo_file).float()
    
    t_ref = _refine_single_tomogram(
        tomo=tomo,
        lightning_model=lightning_model,
        subtomo_size=subtomo_size,
        subtomo_overlap=subtomo_overlap,
        mw_angle=mw_angle,
        normalization_loc=loc,
        normalization_scale=scale,
        num_workers=num_workers,
        batch_size=batch_size,
        pbar_desc=f"Refining {tomo_name}"
    )
    
    return t_ref


# Example usage
if __name__ == "__main__":
    refine_n2v_tomogram(
        tomo_files=["/path/to/your/tomogram.rec"],
        model_checkpoint_file="testing2/logs/version_0/checkpoints/val_loss/epoch=362-val_loss=1.94520.ckpt",
        subtomo_size=128,
        mw_angle=50,
        project_dir="testing_n2v",
        num_workers=0,
        recompute_normalization=False,
        batch_size=10,
        gpu=[0],
    )
