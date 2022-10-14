import blenderproc as bproc

bproc.init()

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass

import dill as pickle
import numpy as np

from causal_images.camera import sample_object_facing_camera_pose
from causal_images.scm import SceneInterventions, SceneManipulations, SceneSCM
from causal_images.util import resolve_object_shapes

# Argument parsing
parser = argparse.ArgumentParser()

parser.add_argument("--output_dir", help="Output directory", required=True)

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

# Scene config
parser.add_argument(
    "--scene_config_path",
)
parser.add_argument(
    "--scene_sampling_config_path",
)

args = parser.parse_args()

scene_conf = {}
if args.scene_config_path is not None:
    with open(args.scene_config_path) as f:
        scene_conf = json.load(f)

scene_sampling_conf = {}
if args.scene_sampling_config_path is not None:
    with open(args.scene_sampling_config_path) as f:
        scene_sampling_conf = json.load(f)

if scene_conf is None and scene_sampling_conf is None:
    raise ValueError(
        "Either scene_config_path or scene_sampling_config_path must be specified."
    )

scene_result = {}

rng = np.random.default_rng(seed=args.seed)

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
    model,
    scene_result,
):
    # Save rendered image
    bproc.writer.write_hdf5(os.path.join(output_dir, str(run_name)), img_data)

    # Save pickle of model
    with open(os.path.join(output_dir, str(run_name), "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open(os.path.join(output_dir, str(run_name), "scene_result.json"), "w") as f:
        json.dump(scene_result, f, cls=NumpyEncoder)

    # Save SCM plot
    fig, ax = model.plot()
    fig.savefig(os.path.join(output_dir, str(run_name), "scm.png"), dpi=fig.dpi)


if "scm" in scene_conf:
    raise NotImplementedError("SCM config not yet supported.")
elif "scm" in scene_sampling_conf:
    scm_conf = scene_sampling_conf["scm"]
    model = None
    if scm_conf["scm_path"] is not None:
        scm = load_module_from_file(scm_conf["scm_path"], "scm")
        model: SceneSCM = scm.scm

    model_interventions = None
    if scm_conf["interventions_path"] is not None:
        interventions = load_module_from_file(
            scm_conf["interventions_path"], "interventions"
        )
        model_interventions: SceneInterventions = interventions.interventions

    model_manipulations = None
    if scm_conf["manipulations_path"] is not None:
        manipulations = load_module_from_file(
            scm_conf["manipulations_path"], "manipulations"
        )
        model_manipulations: SceneManipulations = manipulations.manipulations
else:
    raise ValueError("SCM config not specified.")

light = bproc.types.Light()
position = None
energy = None
if "light" in scene_conf:
    light_conf = scene_conf["light"]
    position = light_conf["position"]
    energy = light_conf["energy"]
elif "light" in scene_sampling_conf:
    light_conf = scene_sampling_conf["light"]
    # TODO: Set BlenderProc seed (https://dlr-rm.github.io/BlenderProc/blenderproc.python.modules.main.InitializerModule.html)
    position = bproc.sampler.shell(
        center=light_conf["center"],
        azimuth_min=light_conf["azimuth_bounds"][0],
        azimuth_max=light_conf["azimuth_bounds"][1],
        elevation_min=light_conf["elevation_bounds"][0],
        elevation_max=light_conf["elevation_bounds"][1],
        radius_min=light_conf["radius_bounds"][0],
        radius_max=light_conf["radius_bounds"][1],
    )
    energy = rng.uniform(*light_conf["energy_bounds"])
else:
    raise ValueError("Light config not specified.")
light.set_location(position)
light.set_energy(energy)
scene_result["light"] = {"position": position, "energy": energy}

for i, df_scene in enumerate(
    model.sample(
        args.scene_num_samples,
        interventions=model_interventions,
        rng=rng,
    )
):
    scene = df_scene.iloc[0]["_scene"]

    # Execute manipulations
    if model_manipulations is not None:
        for (
            node_name,
            manipulation_callable,
        ) in model_manipulations.functional_map_factory(scene, rng).items():
            node_value = df_scene.iloc[0][node_name]
            scene_node_values = df_scene.iloc[0]
            node_value_new = manipulation_callable(node_value, scene_node_values)
            df_scene.iloc[0][node_name] = node_value_new

    df_objects = resolve_object_shapes(df_scene)
    scene_result["scm"] = df_objects.drop(columns=["_scene"]).iloc[0].to_dict()

    objects = [obj.mesh for obj in df_objects.iloc[0]._scene.objects.values()]

    camera_poses = None
    if "camera" in scene_conf:
        camera_poses = np.array(scene_conf["camera"])
    elif "camera" in scene_sampling_conf:
        camera_conf = scene_sampling_conf["camera"]
        camera_poses = np.zeros((camera_conf["num_samples"], 4, 4))
        for j in range(camera_conf["num_samples"]):
            cam2world_matrix = sample_object_facing_camera_pose(
                objects,
                fov_bounds=camera_conf["fov_bounds"],
                camera_zoom_bounds=camera_conf["zoom_bounds"],
                camera_rotation_bounds=camera_conf["rotation_bounds"],
                camera_elevation_bounds=camera_conf["elevation_bounds"],
                camera_azimuth_bounds=camera_conf["azimuth_bounds"],
            )
            camera_poses[j] = cam2world_matrix
    else:
        raise ValueError("No camera configuration found")

    for cam2world_matrix in camera_poses:
        camera = bproc.camera.add_camera_pose(cam2world_matrix)
    scene_result["camera"] = camera_poses

    data = bproc.renderer.render()
    save_outputs(
        output_dir=args.output_dir,
        run_name=i,
        img_data=data,
        model=model,
        scene_result=scene_result,
    )
