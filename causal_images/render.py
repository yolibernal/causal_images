import json

import blenderproc as bproc
import numpy as np

from causal_images.camera import sample_object_facing_camera_pose
from causal_images.scm import SceneInterventions, SceneManipulations, SceneSCM
from causal_images.util import load_module_from_file, save_run_config, save_run_outputs


def create_light(fixed_conf, sampling_conf):
    light = bproc.types.Light()
    position = None
    energy = None
    if "light" in fixed_conf:
        light_conf = fixed_conf["light"]
        position = light_conf["position"]
        energy = light_conf["energy"]
    elif "light" in sampling_conf:
        light_conf = sampling_conf["light"]
        position = bproc.sampler.shell(
            center=light_conf["center"],
            azimuth_min=light_conf["azimuth_bounds"][0],
            azimuth_max=light_conf["azimuth_bounds"][1],
            elevation_min=light_conf["elevation_bounds"][0],
            elevation_max=light_conf["elevation_bounds"][1],
            radius_min=light_conf["radius_bounds"][0],
            radius_max=light_conf["radius_bounds"][1],
        )
        energy = np.random.uniform(*light_conf["energy_bounds"])
    else:
        raise ValueError("Light config not specified.")
    light.set_location(position)
    light.set_energy(energy)
    return light, position, energy


def create_camera_poses(fixed_conf, sampling_conf, objects=None):
    camera_poses = None
    if "camera" in fixed_conf:
        camera_poses = np.array(fixed_conf["camera"])
    elif "camera" in sampling_conf:
        camera_conf = sampling_conf["camera"]
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


def load_model(fixed_conf, sampling_conf):
    if "scm_noise_values" in fixed_conf:
        fixed_noise_values = fixed_conf["scm_noise_values"]

    if "scm_outcomes" in fixed_conf:
        model = SceneSCM.from_scm_outcomes(fixed_conf["scm_outcomes"])
    elif "scm" in sampling_conf:
        scm_conf = sampling_conf["scm"]

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
        if fixed_noise_values is not None:
            model.fixed_noise_values = fixed_noise_values
    else:
        raise ValueError("SCM config not specified.")
    return model


def render_scenes_from_configs(
    fixed_conf, sampling_conf, seed, scene_num_samples, output_dir
):
    if fixed_conf is None and sampling_conf is None:
        raise ValueError("Either fixed_conf or sampling_conf must be specified.")
    if fixed_conf is None:
        fixed_conf = {}
    if sampling_conf is None:
        sampling_conf = {}

    scene_result = {}

    rng = np.random.default_rng(seed=seed)

    light, light_position, light_energy = create_light(fixed_conf, sampling_conf)
    model = load_model(fixed_conf, sampling_conf)

    for i, (scm_outcomes, scm_noise_values, scene) in enumerate(
        model.sample_and_populate_scene(
            scene_num_samples,
            rng=rng,
        )
    ):
        objects = [obj.mesh for obj in scene.objects.values()]
        camera_poses = create_camera_poses(fixed_conf, sampling_conf, objects)
        data = bproc.renderer.render()

        scene_result["scm_outcomes"] = scm_outcomes
        scene_result["scm_noise_values"] = scm_noise_values
        scene_result["camera"] = camera_poses
        scene_result["light"] = {
            "position": light_position,
            "energy": light_energy,
        }
        save_run_outputs(
            output_dir=output_dir,
            run_name=i,
            img_data=data,
            scene_result=scene_result,
        )
    save_run_config(
        output_dir=output_dir,
        model=model,
        fixed_conf=fixed_conf,
        sampling_conf=sampling_conf,
    )
