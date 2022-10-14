import blenderproc as bproc

bproc.init()

import argparse
import importlib
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass

import dill as pickle
import numpy as np
import pandas as pd

from causal_images.camera import sample_object_facing_camera_pose
from causal_images.scm import SceneInterventions, SceneManipulations, SceneSCM
from causal_images.util import resolve_object_shapes

# Argument parsing
parser = argparse.ArgumentParser()

parser.add_argument("--output_dir", help="Output directory", required=True)
parser.add_argument(
    "--scm_path",
    help="Path to Structural Causal Model (SCM) Python file. Should declare instance of causal_images.scm.SceneSCM as `scm`.",
    required=True,
)
parser.add_argument(
    "--interventions_path",
    help="Path to Intervention Python file. Should declare instance of causal_images.scm.SceneInterventions as `interventions`.",
)

parser.add_argument(
    "--manipulations_path",
    help="Path to Manipulations Python file. Should declare instance of causal_images.scm.SceneManipulations as `manipulations`.",
)

parser.add_argument(
    "--seed",
    help="Seed for initializing random generator.",
    type=int,
)

parser.add_argument(
    "--scene_num_samples",
    default=5,
    type=int,
    help="Number of sampled scenes",
)

# Camera arguments
parser.add_argument(
    "--camera_num_samples",
    default=5,
    type=int,
    help="Number of camera samples per scene",
)

parser.add_argument(
    "--camera_fov_lower",
    help="Lower level for field of view sampling",
    default=np.pi / 6,
    type=float,
)
parser.add_argument(
    "--camera_fov_upper",
    help="Upper level for field of view sampling",
    default=np.pi / 6,
    type=float,
)
parser.add_argument(
    "--camera_zoom_lower", help="Lower bound for zoom sampling", default=1, type=float
)
parser.add_argument(
    "--camera_zoom_upper", help="Upper bound for zoom sampling", default=1, type=float
)
parser.add_argument(
    "--camera_azimuth_lower",
    help="Lower bound for azimuth sampling",
    default=-1,
    type=float,
)
parser.add_argument(
    "--camera_azimuth_upper",
    help="Upper bound for azimuth sampling",
    default=1,
    type=float,
)
parser.add_argument(
    "--camera_elevation_lower",
    help="Lower bound for elevation sampling",
    default=-1,
    type=float,
)
parser.add_argument(
    "--camera_elevation_upper",
    help="Upper bound for elevation sampling",
    default=1,
    type=float,
)
parser.add_argument(
    "--camera_rotation_lower",
    default=0,
    type=float,
    help="Lower bound for camera rotation (pi)",
)
parser.add_argument(
    "--camera_rotation_upper",
    default=0,
    type=float,
    help="Upper bound for camera rotation (pi)",
)

# Light arguments
parser.add_argument(
    "--light_position",
    nargs=3,
    default=[2, -2, 0],
    type=float,
    help="Light position",
)
parser.add_argument(
    "--light_energy",
    default=500,
    type=float,
    help="Light energy value",
)

args = parser.parse_args()


@dataclass
class LightConfig:
    position: np.ndarray
    energy: float


# https://stackoverflow.com/a/67692
def load_module_from_file(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def save_outputs(
    output_dir,
    run_name,
    img_data,
    df_objects,
    model,
    scm_path,
    interventions_path,
    manipulations_path,
    camera_poses: np.ndarray,
    light_config,
):
    # Save rendered image
    bproc.writer.write_hdf5(os.path.join(output_dir, str(run_name)), img_data)
    # Save sampled values
    df_objects.drop(columns=["_scene"]).to_csv(
        os.path.join(output_dir, str(run_name), "result.csv"), index=False
    )
    # Save SCM Python definition
    shutil.copyfile(scm_path, os.path.join(output_dir, str(run_name), "scm.py"))
    if interventions_path is not None:
        shutil.copyfile(
            interventions_path,
            os.path.join(output_dir, str(run_name), "interventions.py"),
        )
    if manipulations_path is not None:
        shutil.copyfile(
            manipulations_path,
            os.path.join(output_dir, str(run_name), "manipulations.py"),
        )
    # Save camera poses
    for i, camera_pose in enumerate(camera_poses):
        np.savetxt(
            os.path.join(output_dir, str(run_name), f"{i}_camera_pose.txt"), camera_pose
        )
    # Save pickle of model
    with open(os.path.join(output_dir, str(run_name), "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(output_dir, str(run_name), "light_config.json"), "w") as f:
        json.dump(asdict(light_config), f)

    # Save SCM plot
    fig, ax = model.plot()
    fig.savefig(os.path.join(output_dir, str(run_name), "scm.png"), dpi=fig.dpi)


scm = load_module_from_file(args.scm_path, "scm")
model: SceneSCM = scm.scm

model_interventions = None
if args.interventions_path:
    interventions = load_module_from_file(args.interventions_path, "interventions")
    model_interventions: SceneInterventions = interventions.interventions

model_manipulations = None
if args.manipulations_path:
    manipulations = load_module_from_file(args.manipulations_path, "manipulations")
    model_manipulations: SceneManipulations = manipulations.manipulations

light_config = LightConfig(position=args.light_position, energy=args.light_energy)
light = bproc.types.Light()
light.set_location(light_config.position)
light.set_energy(light_config.energy)

rng = np.random.default_rng(seed=args.seed)

for i, df_scene in enumerate(
    model.sample(
        args.scene_num_samples,
        interventions=model_interventions,
        rng=rng,
    )
):
    scene = df_scene.iloc[0]["_scene"]

    # Execute manipulations
    for node_name, manipulation_callable in model_manipulations.functional_map_factory(
        scene, rng
    ).items():
        node_value = df_scene.iloc[0][node_name]
        scene_node_values = df_scene.iloc[0]
        node_value_new = manipulation_callable(node_value, scene_node_values)
        df_scene.iloc[0][node_name] = node_value_new

    df_objects = resolve_object_shapes(df_scene)

    objects = [obj.mesh for obj in df_objects.iloc[0]._scene.objects.values()]

    camera_poses = np.zeros((args.camera_num_samples, 4, 4))
    for j in range(args.camera_num_samples):
        cam2world_matrix = sample_object_facing_camera_pose(
            objects,
            fov_bounds=(args.camera_fov_lower, args.camera_fov_upper),
            camera_zoom_bounds=(args.camera_zoom_lower, args.camera_zoom_upper),
            camera_rotation_bounds=(
                args.camera_rotation_lower,
                args.camera_rotation_upper,
            ),
            camera_elevation_bounds=(
                args.camera_elevation_lower,
                args.camera_elevation_upper,
            ),
            camera_azimuth_bounds=(
                args.camera_azimuth_lower,
                args.camera_azimuth_upper,
            ),
        )
        camera_poses[j] = cam2world_matrix
        camera = bproc.camera.add_camera_pose(cam2world_matrix)

    data = bproc.renderer.render()
    save_outputs(
        output_dir=args.output_dir,
        run_name=i,
        img_data=data,
        df_objects=df_objects,
        model=model,
        scm_path=args.scm_path,
        interventions_path=args.interventions_path,
        manipulations_path=args.manipulations_path,
        camera_poses=camera_poses,
        light_config=light_config,
    )
