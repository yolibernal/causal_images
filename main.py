import blenderproc as bproc
import numpy as np

bproc.init()

import glob
import json
import os
import shutil

import click

from causal_images.render import render_scenes_from_configs
from causal_images.util import hdf5_to_image

ALLOW_COLLISIONS = True
ENABLE_TRANSPARENCY = True


@click.group()
def cli():
    pass


@cli.command()
@click.option("--fixed_config_path", "--fixed_config", help="Fixed config")
@click.option("--sampling_config_path", "--sampling_config", help="Sampling config")
@click.option("--material_library_path", help="Path to .blend file containing materials")
@click.option("--output_dir", help="Output directory", required=True)
@click.option("--seed", help="Seed for initializing random generator.", type=int)
@click.option("--scene_num_samples", default=5, type=int, help="Number of sampled scenes")
@click.option("--skip_render", help="Skip rendering", is_flag=True)
def sample(
    fixed_config_path,
    sampling_config_path,
    material_library_path,
    output_dir,
    seed,
    scene_num_samples,
    batch_size=200,
    skip_render=False,
):
    fixed_conf = None
    if fixed_config_path is not None:
        with open(fixed_config_path) as f:
            fixed_conf = json.load(f)

    sampling_conf = None
    if sampling_config_path is not None:
        with open(sampling_config_path) as f:
            sampling_conf = json.load(f)

    if fixed_conf is None and sampling_conf is None:
        raise ValueError("Either fixed_config or sampling_config must be specified.")

    if seed is not None:
        os.environ["BLENDER_PROC_RANDOM_SEED"] = seed

    rng = np.random.default_rng(seed=seed)

    if batch_size is not None:
        num_batches = int(np.ceil(scene_num_samples / batch_size))
        current_batch_size = batch_size
        for i in range(num_batches):
            if i == num_batches - 1:
                current_batch_size = scene_num_samples - i * batch_size

            batch_seed = rng.integers(0, 2**32 - 1)
            render_scenes_from_configs(
                fixed_conf=fixed_conf,
                sampling_conf=sampling_conf,
                material_library_path=material_library_path,
                seed=batch_seed,
                scene_num_samples=current_batch_size,
                output_dir=output_dir,
                run_names=range(i * batch_size, (i + 1) * batch_size),
                allow_collisions=ALLOW_COLLISIONS,
                enable_transparency=ENABLE_TRANSPARENCY,
                skip_render=skip_render,
            )
            bproc.clean_up()

    else:
        render_scenes_from_configs(
            fixed_conf=fixed_conf,
            sampling_conf=sampling_conf,
            material_library_path=material_library_path,
            seed=seed,
            scene_num_samples=scene_num_samples,
            output_dir=output_dir,
            allow_collisions=ALLOW_COLLISIONS,
            enable_transparency=ENABLE_TRANSPARENCY,
        )
        bproc.clean_up()


def interpolate_dict_values(dict1, dict2, alpha, interpolation="linear"):
    if interpolation == "linear":
        return {k: [(1 - alpha) * dict1[k][0] + alpha * dict2[k][0]] for k in dict1}
    else:
        raise ValueError(f"Interpolation method {interpolation} not supported.")


@cli.command()
@click.option("--input_dir", help="Input directory", required=True)
@click.option("--output_dir", help="Output directory", required=True)
@click.option("--sequence_output_dir", help="Sequence output directory", required=True)
@click.option("--material_library_path", help="Path to .blend file containing materials")
@click.option("--interventions_path", help="Interventions file", required=True)
@click.option("--seed", help="Seed for initializing random generator.", type=int)
@click.option("--sequence_length", help="Sequence length", type=int, default=1)
@click.option("--skip_render", help="Skip rendering", is_flag=True)
def counterfactual(
    input_dir,
    output_dir,
    sequence_output_dir,
    material_library_path,
    interventions_path,
    seed,
    sequence_length: int = 1,
    skip_render=False,
):
    # Load the scm
    scm_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if "_SCM_" in f and f.endswith(".py")
    ]
    if len(scm_paths) != 1:
        raise ValueError(
            f"Expected exactly one SCM file. SCM paths: {scm_paths}, Input dir: {input_dir}"
        )
    scm_path = scm_paths[0]
    # scm_module = load_module_from_file(scm_path, "scm")
    # model = scm_module.scm

    sampling_conf_counterfactual = {
        "scm": {
            "scm_path": scm_path,
            "interventions_path": interventions_path,
            "manipulations_path": None,
        }
    }
    excluded_dirs = ["__pycache__"]
    run_directories = next(os.walk(input_dir))[1]

    if seed is not None:
        os.environ["BLENDER_PROC_RANDOM_SEED"] = seed
    rng = np.random.default_rng(seed=seed)

    for run_dir in run_directories:
        if run_dir in excluded_dirs:
            continue
        scene_result_path = os.path.join(input_dir, run_dir, "scene_result.json")
        if not os.path.exists(scene_result_path):
            raise ValueError(f"Scene result not found for {run_dir}.")
        with open(scene_result_path) as f:
            scene_result = json.load(f)
        # Use original scene result to sample counterfactual (remove outcomes to allow resampling of intervened values)
        fixed_conf_counterfactual = {
            k: v for k, v in scene_result.items() if k not in ["scm_outcomes"]
        }
        run_seed = rng.integers(0, 2**32 - 1)
        render_scenes_from_configs(
            fixed_conf=fixed_conf_counterfactual,
            sampling_conf=sampling_conf_counterfactual,
            material_library_path=material_library_path,
            seed=run_seed,
            scene_num_samples=1,
            output_dir=os.path.join(output_dir, run_dir),
            allow_collisions=ALLOW_COLLISIONS,
            enable_transparency=ENABLE_TRANSPARENCY,
            skip_render=skip_render,
        )

        if sequence_length > 1:
            fixed_conf_sequence = fixed_conf_counterfactual.copy()
            sampling_conf_sequence = {
                "scm": {
                    "scm_path": scm_path,
                    "interventions_path": None,
                    "manipulations_path": None,
                }
            }
            if sequence_output_dir is None:
                raise ValueError(
                    "Sequence output directory must be specified for sequence rendering."
                )
            scene_result_counterfactual_file = os.path.join(
                output_dir, run_dir, "0", "scene_result.json"
            )
            with open(scene_result_counterfactual_file) as f:
                scene_result_counterfactual = json.load(f)

            # interpolate between original and counterfactual noise values
            noise_values = scene_result["scm_noise_values"]
            noise_values_counterfactual = scene_result_counterfactual["scm_noise_values"]
            alphas = np.linspace(0, 1, sequence_length + 1)[1:]
            for i in range(sequence_length):
                interpolated_noise_values = interpolate_dict_values(
                    noise_values, noise_values_counterfactual, alphas[i], interpolation="linear"
                )
                fixed_conf_sequence["scm_noise_values"] = interpolated_noise_values
                render_scenes_from_configs(
                    fixed_conf=fixed_conf_sequence,
                    sampling_conf=sampling_conf_sequence,
                    material_library_path=material_library_path,
                    seed=run_seed,
                    scene_num_samples=1,
                    run_names=[i],
                    output_dir=os.path.join(sequence_output_dir, run_dir),
                    allow_collisions=ALLOW_COLLISIONS,
                    enable_transparency=ENABLE_TRANSPARENCY,
                    skip_render=skip_render,
                )


@cli.command()
@click.option("--input_dir", help="Input directory", required=True)
@click.option("--output_dir", help="Output directory", required=True)
@click.option("--to_image", help="Convert hdf5 to image", is_flag=True)
@click.option("--sequence_length", help="Sequence length", type=int, default=1)
@click.option("--image_format", help="Image format", default="JPEG")
@click.option("--skip_image", help="Skip rendering", is_flag=True)
def export(input_dir, output_dir, to_image, sequence_length, image_format: str, skip_image=False):
    file_index = 0

    output_data_dir = os.path.join(output_dir, "data")
    output_labels_dir = os.path.join(output_dir, "labels")
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            resultspath = os.path.join(subdir, file)
            if resultspath.endswith("scene_result.json"):
                dir_path = os.path.dirname(resultspath)

                filepath = os.path.join(dir_path, "0.hdf5")

                if sequence_length > 1:
                    # Use parent directory as sequence index
                    filename = f"{file_index // sequence_length}_{dir_path.split(os.path.sep)[-1]}"
                else:
                    filename = f"{file_index}"
                scene_result_path = os.path.join(dir_path, "scene_result.json")

                if not skip_image:
                    if to_image:
                        img = hdf5_to_image(filepath)
                        img.save(
                            os.path.join(output_data_dir, f"{filename}.{image_format.lower()}")
                        )
                    else:
                        shutil.copy(filepath, os.path.join(output_data_dir, f"{filename}.hdf5"))

                with open(scene_result_path) as f:
                    scene_result = json.load(f)

                # NOTE: Not saving cameras because they are not specific for only that image (camera might be list of multiple sampled poses)
                label = {
                    k: v
                    for k, v in scene_result.items()
                    if k in ["scm_outcomes", "scm_noise_values"]
                }
                with open(os.path.join(output_labels_dir, f"{filename}.json"), "w") as f:
                    json.dump(label, f)
                file_index += 1


@cli.command()
@click.option("--fixed_config_path", "--fixed_config", help="Fixed config")
@click.option("--sampling_config_path", "--sampling_config", help="Sampling config")
@click.option("--output_dir", help="Output directory", required=True)
@click.option("--interventions_dir", help="Interventions directory", required=True)
@click.option("--material_library_path", help="Path to .blend file containing materials")
@click.option("--num_samples", default=5, type=int, help="Number of sampled scenes")
@click.option(
    "--intervention_probabilities",
    default=[-1],
    type=float,
    multiple=True,
    help="Number of sampled scenes per intervention",
)
@click.option(
    "--seed",
    help="Seed for initializing random generator.",
    type=int,
)
@click.option("--output_image_dir", help="Image output directory", required=True)
@click.option("--sequence_length", help="Sequence length", type=int, default=1)
@click.option(
    "--to_image",
    help="Convert hdf5 to image",
    is_flag=True,
)
@click.option("--image_format", help="Image format", default="JPEG")
@click.option("--skip_render", help="Skip rendering", is_flag=True)
@click.pass_context
def sample_weakly_supervised(
    ctx,
    fixed_config_path,
    sampling_config_path,
    output_dir,
    interventions_dir,
    material_library_path,
    num_samples,
    intervention_probabilities,
    seed,
    output_image_dir,
    to_image,
    sequence_length: int,
    image_format: str,
    skip_render=False,
):
    print("Sampling original and counterfactual scenes...")
    intervention_paths = [
        os.path.join(interventions_dir, f)
        for f in os.listdir(interventions_dir)
        if f.endswith(".py")
    ]
    intervention_paths = sorted(intervention_paths, key=lambda x: os.path.basename(x))
    EMPTY_INTERVENTION = "_empty_intervention"
    intervention_paths_with_empty = [EMPTY_INTERVENTION] + intervention_paths

    if len(intervention_probabilities) != len(
        intervention_paths_with_empty
    ) and intervention_probabilities != (-1,):
        raise ValueError(
            f"Number of intervention probabilities must match number of interventions or be (-1,) for uniform."
        )

    if intervention_probabilities == (-1,):
        intervention_probabilities = [1 / len(intervention_paths_with_empty)] * len(
            intervention_paths_with_empty
        )

    # Sample number of scenes per intervention
    num_samples_per_intervention = np.random.multinomial(
        num_samples, intervention_probabilities
    ).tolist()

    for num_samples in num_samples_per_intervention:
        if num_samples == 0:
            raise ValueError("Number of samples per intervention cannot be 0.")

    print("Interventions paths:", intervention_paths)
    for interventions_path, intervention_num_samples in zip(
        intervention_paths_with_empty, num_samples_per_intervention
    ):
        print("Sampling for interventions path:", interventions_path)
        intervention_name = (
            os.path.basename(interventions_path).replace(".py", "")
            if interventions_path != EMPTY_INTERVENTION
            else EMPTY_INTERVENTION
        )
        intervention_output_dir = os.path.join(output_dir, intervention_name)

        original_output_dir = os.path.join(intervention_output_dir, "original")
        if not os.path.exists(original_output_dir):
            os.makedirs(original_output_dir)

        counterfactual_output_dir = os.path.join(intervention_output_dir, "counterfactual")
        if not os.path.exists(counterfactual_output_dir):
            os.makedirs(counterfactual_output_dir)

        if sequence_length > 1:
            sequence_output_dir = os.path.join(intervention_output_dir, "sequence")
            if not os.path.exists(sequence_output_dir):
                os.makedirs(sequence_output_dir)

        # Sample original scenes
        ctx.invoke(
            sample,
            fixed_config_path=fixed_config_path,
            sampling_config_path=sampling_config_path,
            material_library_path=material_library_path,
            seed=seed,
            output_dir=original_output_dir,
            scene_num_samples=intervention_num_samples,
            skip_render=skip_render,
        )

        # If empty intervention, we don't need counterfactual and can use the original scenes instead
        if intervention_name != EMPTY_INTERVENTION:
            # Sample counterfactual scenes
            ctx.invoke(
                counterfactual,
                input_dir=original_output_dir,
                output_dir=counterfactual_output_dir,
                sequence_output_dir=sequence_output_dir,
                material_library_path=material_library_path,
                interventions_path=interventions_path,
                sequence_length=sequence_length,
                seed=seed,
                skip_render=skip_render,
            )
        if to_image:
            # Create image directories
            original_image_dir = os.path.join(output_image_dir, intervention_name, "original")
            if not os.path.exists(original_image_dir):
                os.makedirs(original_image_dir)
            counterfactual_image_dir = os.path.join(
                output_image_dir, intervention_name, "counterfactual"
            )
            if not os.path.exists(counterfactual_image_dir):
                os.makedirs(counterfactual_image_dir)

            # export originals
            ctx.invoke(
                export,
                input_dir=original_output_dir,
                output_dir=original_image_dir,
                to_image=True,
                image_format=image_format,
                skip_image=skip_render,
            )
            # export counterfactuals
            ctx.invoke(
                export,
                # if empty intervention just copy the original images
                input_dir=(
                    counterfactual_output_dir
                    if intervention_name != EMPTY_INTERVENTION
                    else original_output_dir
                ),
                output_dir=counterfactual_image_dir,
                to_image=True,
                image_format=image_format,
                skip_image=skip_render,
            )

            # export sequences
            if sequence_length > 1:
                sequence_image_dir = os.path.join(output_image_dir, intervention_name, "sequence")
                if not os.path.exists(sequence_image_dir):
                    os.makedirs(sequence_image_dir)
                ctx.invoke(
                    export,
                    input_dir=sequence_output_dir,
                    output_dir=sequence_image_dir,
                    to_image=True,
                    sequence_length=sequence_length,
                    image_format=image_format,
                    skip_image=skip_render,
                )

            # if empty intervention, copy original image sequence_length times
            if intervention_name == EMPTY_INTERVENTION:
                for i in range(intervention_num_samples):
                    for j in range(sequence_length):
                        # Get a list of all image files in the original_image_dir
                        image_files = glob.glob(os.path.join(original_image_dir, "data", f"{i}.*"))

                        label_files = glob.glob(
                            os.path.join(original_image_dir, "labels", f"{i}.*")
                        )

                        # Iterate over each image file and copy it to the sequence_image_dir
                        if not skip_render:
                            for image_file in image_files:
                                assert (
                                    len(image_files) > 0
                                ), f"No images found in {original_image_dir}"
                                # Extract the file extension
                                _, extension = os.path.splitext(image_file)

                                # Define the new filename in the sequence_image_dir
                                new_filename = (
                                    f"{i}_{j}{extension}"
                                    if sequence_length > 1
                                    else f"{i}{extension}"
                                )

                                # Copy the image file to the sequence_image_dir
                                shutil.copy(
                                    image_file,
                                    os.path.join(sequence_image_dir, "data", new_filename),
                                )
                        for label_file in label_files:
                            assert len(label_files) > 0, f"No labels found in {original_image_dir}"
                            # Extract the file extension
                            _, extension = os.path.splitext(label_file)

                            # Define the new filename in the sequence_image_dir
                            new_filename = (
                                f"{i}_{j}{extension}" if sequence_length > 1 else f"{i}{extension}"
                            )

                            # Copy the image file to the sequence_image_dir
                            shutil.copy(
                                label_file,
                                os.path.join(sequence_image_dir, "labels", new_filename),
                            )


if __name__ == "__main__":
    cli()
