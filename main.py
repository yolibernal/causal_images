import blenderproc as bproc

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
@click.option("--output_dir", help="Output directory", required=True)
@click.option(
    "--seed",
    help="Seed for initializing random generator.",
    type=int,
)
@click.option(
    "--scene_num_samples",
    default=5,
    type=int,
    help="Number of sampled scenes",
)
def sample(fixed_config_path, sampling_config_path, output_dir, seed, scene_num_samples):
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

    render_scenes_from_configs(
        fixed_conf=fixed_conf,
        sampling_conf=sampling_conf,
        seed=seed,
        scene_num_samples=scene_num_samples,
        output_dir=output_dir,
    )


@cli.command()
@click.option("--input_dir", help="Input directory", required=True)
@click.option("--output_dir", help="Output directory", required=True)
@click.option("--interventions_path", help="Interventions file", required=True)
@click.option(
    "--seed",
    help="Seed for initializing random generator.",
    type=int,
    default=0,
)
def counterfactual(input_dir, output_dir, interventions_path, seed):
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
    for run_dir in run_directories:
        if run_dir in excluded_dirs:
            continue
        scene_result_path = os.path.join(input_dir, run_dir, "scene_result.json")
        if not os.path.exists(scene_result_path):
            raise ValueError(f"Scene result not found for {run_dir}.")
        with open(scene_result_path) as f:
            scene_result = json.load(f)
        fixed_conf = {k: v for k, v in scene_result.items() if k not in ["scm_outcomes"]}
        render_scenes_from_configs(
            fixed_conf=fixed_conf,
            sampling_conf=sampling_conf,
            seed=seed,
            scene_num_samples=1,
            output_dir=os.path.join(output_dir, run_dir),
        )


@cli.command()
@click.option("--input_dir", help="Input directory", required=True)
@click.option("--output_dir", help="Output directory", required=True)
@click.option("--to-image", help="Convert hdf5 to image", is_flag=True)
def export(input_dir, output_dir, to_image):
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
                    hdf5_to_image(filepath, os.path.join(output_data_dir, f"{file_index}.jpg"))
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


if __name__ == "__main__":
    cli()
