import blenderproc as bproc

bproc.init()

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys

import dill as pickle
import numpy as np

from causal_images.camera import sample_object_facing_camera_pose
from causal_images.scene import PrimitiveShape
from causal_images.scm import SceneInterventions, SceneManipulations, SceneSCM
from causal_images.util import resolve_object_shapes

# TODO: Add hierarchical configs
# TODO: Merge defaults
# TODO: Set BlenderProc seed (https://dlr-rm.github.io/BlenderProc/blenderproc.python.modules.main.InitializerModule.html)

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

# TODO: allow multiple interventions and manipulations
scene_sampling_conf = None
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


def save_model_component_file(
    output_dir, run_name, component_path: str, component_type: str
):
    component_name = os.path.split(component_path)[1].split(".")[0]
    with open(component_path) as f:
        file_hash = hashlib.md5(f.read().encode("utf-8")).hexdigest()
    shutil.copyfile(
        component_path,
        os.path.join(
            output_dir,
            str(run_name),
            f"{component_name}_{component_type.upper()}_{file_hash}.py",
        ),
    )


def save_outputs(
    output_dir, run_name, img_data, model, scene_result, scene_conf, scene_sampling_conf
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

    if scene_conf is not None:
        with open(
            os.path.join(output_dir, str(run_name), "scene_config.json"), "w"
        ) as f:
            json.dump(scene_conf, f, cls=NumpyEncoder)

    if scene_sampling_conf is not None:
        with open(
            os.path.join(output_dir, str(run_name), "scene_sampling_config.json"), "w"
        ) as f:
            json.dump(scene_sampling_conf, f, cls=NumpyEncoder)

        if scene_sampling_conf["scm"] is not None:
            scm_conf = scene_sampling_conf["scm"]

            if scm_conf["scm_path"] is not None:
                save_model_component_file(
                    output_dir, run_name, scm_conf["scm_path"], "scm"
                )
            if scm_conf["interventions_path"] is not None:
                save_model_component_file(
                    output_dir,
                    run_name,
                    scm_conf["interventions_path"],
                    "interventions",
                )
            if scm_conf["manipulations_path"] is not None:
                save_model_component_file(
                    output_dir,
                    run_name,
                    scm_conf["manipulations_path"],
                    "manipulations",
                )


def create_deterministic_node_callable(scene, node_name: str, node_value):
    """Create a callable that returns a constant node value."""
    if node_name.startswith("obj_"):
        return (
            [],
            lambda: scene.create_primitive(PrimitiveShape(node_value)),
            None,
        )
    elif node_name.startswith("pos_"):
        return (
            [node_name.replace("pos_", "obj_")],
            lambda obj_parent: scene.set_object_position(obj_parent, node_value),
            None,
        )
    else:
        return ([], lambda: node_value, None)


def create_deterministic_scm(scene_conf):
    model = SceneSCM(
        lambda scene, rng: {
            node_name: create_deterministic_node_callable(scene, node_name, node_value)
            for node_name, node_value in scene_conf["scm"].items()
        }
    )
    return model


def create_scm(scene_sampling_conf):
    scm_conf = scene_sampling_conf["scm"]

    # if scm_conf["scm"] is not None:
    scm = load_module_from_file(scm_conf["scm_path"], "scm")
    model: SceneSCM = scm.scm

    if scm_conf["interventions_path"] is not None:
        interventions = load_module_from_file(
            scm_conf["interventions_path"], "interventions"
        )
        model_interventions: SceneInterventions = interventions.interventions
        model.interventions = model_interventions

    if scm_conf["manipulations_path"] is not None:
        manipulations = load_module_from_file(
            scm_conf["manipulations_path"], "manipulations"
        )
        model_manipulations: SceneManipulations = manipulations.manipulations
        model.manipulations = model_manipulations
    return model


def create_model(scene_conf, scene_sampling_conf):
    if "scm" in scene_conf:
        model = create_deterministic_scm(scene_conf)
    elif "scm" in scene_sampling_conf:
        model = create_scm(scene_sampling_conf)
    else:
        raise ValueError("SCM config not specified.")
    return model


def create_light(scene_conf, scene_sampling_conf):
    light = bproc.types.Light()
    position = None
    energy = None
    if "light" in scene_conf:
        light_conf = scene_conf["light"]
        position = light_conf["position"]
        energy = light_conf["energy"]
    elif "light" in scene_sampling_conf:
        light_conf = scene_sampling_conf["light"]
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
    return light, position, energy


def create_camera_poses(scene_conf, scene_sampling_conf, objects=None):
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
        raise ValueError("Camera config not specified.")
    for cam2world_matrix in camera_poses:
        bproc.camera.add_camera_pose(cam2world_matrix)
    return camera_poses


light, light_position, light_energy = create_light(scene_conf, scene_sampling_conf)
model = create_model(scene_conf, scene_sampling_conf)

for i, df_scene in enumerate(
    model.sample_and_populate_scene(
        args.scene_num_samples,
        rng=rng,
    )
):
    objects = [obj.mesh for obj in df_scene.iloc[0]._scene.objects.values()]
    camera_poses = create_camera_poses(scene_conf, scene_sampling_conf, objects)
    data = bproc.renderer.render()

    scene_result["scm_outcomes"] = df_scene.drop(columns=["_scene"]).iloc[0].to_dict()
    scene_result["camera"] = camera_poses
    scene_result["light"] = {
        "position": light_position,
        "energy": light_energy,
    }
    save_outputs(
        output_dir=args.output_dir,
        run_name=i,
        img_data=data,
        model=model,
        scene_result=scene_result,
        scene_conf=scene_conf,
        scene_sampling_conf=scene_sampling_conf,
    )
