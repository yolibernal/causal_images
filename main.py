import blenderproc as bproc
import numpy as np

bproc.init()

import json
import os
import shutil

import click

from causal_images.render import render_scenes_from_configs
from causal_images.util import hdf5_to_image


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
def sample(
    fixed_config_path,
    sampling_config_path,
    material_library_path,
    output_dir,
    seed,
    scene_num_samples,
    batch_size=200,
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
        )
        bproc.clean_up()


@cli.command()
@click.option("--input_dir", help="Input directory", required=True)
@click.option("--output_dir", help="Output directory", required=True)
@click.option("--material_library_path", help="Path to .blend file containing materials")
@click.option("--interventions_path", help="Interventions file", required=True)
@click.option("--seed", help="Seed for initializing random generator.", type=int)
def counterfactual(input_dir, output_dir, material_library_path, interventions_path, seed):
    # Load the scm
    scm_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if "_SCM_" in f and f.endswith(".py")
    ]
    if len(scm_paths) != 1:
        raise ValueError("Expected exactly one SCM file.")
    scm_path = scm_paths[0]
    # scm_module = load_module_from_file(scm_path, "scm")
    # model = scm_module.scm

    sampling_conf = {
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
        fixed_conf = {k: v for k, v in scene_result.items() if k not in ["scm_outcomes"]}
        run_seed = rng.integers(0, 2**32 - 1)
        render_scenes_from_configs(
            fixed_conf=fixed_conf,
            sampling_conf=sampling_conf,
            material_library_path=material_library_path,
            seed=run_seed,
            scene_num_samples=1,
            output_dir=os.path.join(output_dir, run_dir),
        )


@cli.command()
@click.option("--input_dir", help="Input directory", required=True)
@click.option("--output_dir", help="Output directory", required=True)
@click.option("--to_image", help="Convert hdf5 to image", is_flag=True)
@click.option("--image_format", help="Image format", default="JPEG")
def export(input_dir, output_dir, to_image, image_format: str):
    file_index = 0

    output_data_dir = os.path.join(output_dir, "data")
    output_labels_dir = os.path.join(output_dir, "labels")
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith(".hdf5"):
                dir_path = os.path.dirname(filepath)
                scene_result_path = os.path.join(dir_path, "scene_result.json")
                if to_image:
                    img = hdf5_to_image(filepath)
                    img.save(os.path.join(output_data_dir, f"{file_index}.{image_format.lower()}"))
                else:
                    shutil.copy(filepath, os.path.join(output_data_dir, f"{file_index}.hdf5"))
                with open(scene_result_path) as f:
                    scene_result = json.load(f)

                # NOTE: Not saving cameras because they are not specific for only that image (camera might be list of multiple sampled poses)
                label = {
                    k: v
                    for k, v in scene_result.items()
                    if k in ["scm_outcomes", "scm_noise_values"]
                }
                with open(os.path.join(output_labels_dir, f"{file_index}.json"), "w") as f:
                    json.dump(label, f)
                file_index += 1


@cli.command()
@click.option("--fixed_config_path", "--fixed_config", help="Fixed config")
@click.option("--sampling_config_path", "--sampling_config", help="Sampling config")
@click.option("--output_dir", help="Output directory", required=True)
@click.option("--interventions_dir", help="Interventions directory", required=True)
@click.option("--material_library_path", help="Path to .blend file containing materials")
@click.option(
    "--num_samples_per_intervention",
    default=[5],
    type=int,
    multiple=True,
    help="Number of sampled scenes per intervention",
)
@click.option(
    "--seed",
    help="Seed for initializing random generator.",
    type=int,
)
@click.option("--output_image_dir", help="Image output directory", required=True)
@click.option(
    "--to_image",
    help="Convert hdf5 to image",
    is_flag=True,
)
@click.option("--image_format", help="Image format", default="JPEG")
@click.pass_context
def sample_weakly_supervised(
    ctx,
    fixed_config_path,
    sampling_config_path,
    output_dir,
    interventions_dir,
    material_library_path,
    num_samples_per_intervention,
    seed,
    output_image_dir,
    to_image,
    image_format: str,
):
    print("Sampling original and counterfactual scenes...")
    interventions_paths = [
        os.path.join(interventions_dir, f)
        for f in os.listdir(interventions_dir)
        if f.endswith(".py")
    ]

    if (
        len(num_samples_per_intervention) != len(interventions_paths)
        and len(num_samples_per_intervention) != 1
    ):
        raise ValueError(
            "Length of number of samples per intervention must be either 1 or equal to the number of interventions."
        )

    if len(num_samples_per_intervention) == 1:
        num_samples_per_intervention = num_samples_per_intervention * len(interventions_paths)

    EMPTY_INTERVENTION = "_empty_intervention"
    print("Interventions paths:", interventions_paths)
    for i, interventions_path in enumerate(interventions_paths + [EMPTY_INTERVENTION]):
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

        ctx.invoke(
            sample,
            fixed_config_path=fixed_config_path,
            sampling_config_path=sampling_config_path,
            material_library_path=material_library_path,
            seed=seed,
            output_dir=original_output_dir,
            scene_num_samples=num_samples_per_intervention[i],
        )
        if intervention_name != EMPTY_INTERVENTION:
            ctx.invoke(
                counterfactual,
                input_dir=original_output_dir,
                output_dir=counterfactual_output_dir,
                material_library_path=material_library_path,
                interventions_path=interventions_path,
                seed=seed,
            )
        if to_image:
            original_image_dir = os.path.join(output_image_dir, intervention_name, "original")
            if not os.path.exists(original_image_dir):
                os.makedirs(original_image_dir)
            counterfactual_image_dir = os.path.join(
                output_image_dir, intervention_name, "counterfactual"
            )
            if not os.path.exists(counterfactual_image_dir):
                os.makedirs(counterfactual_image_dir)
            ctx.invoke(
                export,
                input_dir=original_output_dir,
                output_dir=original_image_dir,
                to_image=True,
                image_format=image_format,
            )
            ctx.invoke(
                export,
                input_dir=counterfactual_output_dir
                if intervention_name != EMPTY_INTERVENTION
                else original_output_dir,
                output_dir=counterfactual_image_dir,
                to_image=True,
                image_format=image_format,
            )


if __name__ == "__main__":
    cli()
