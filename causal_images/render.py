import blenderproc as bproc
import numpy as np

from causal_images.camera import sample_object_facing_camera_pose
from causal_images.scm import SceneInterventions, SceneManipulations, SceneSCM
from causal_images.util import load_module_from_file, save_outputs


def create_light(scene_conf, scene_sampling_conf, rng=np.random.default_rng()):
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


def load_model(scene_conf, scene_sampling_conf):
    if "scm_outcomes" in scene_conf:
        model = SceneSCM.from_scm_outcomes(scene_conf["scm_outcomes"])
    elif "scm" in scene_sampling_conf:
        scm_conf = scene_sampling_conf["scm"]

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
    else:
        raise ValueError("SCM config not specified.")
    return model


def render_scenes(args, scene_conf, scene_sampling_conf):
    if scene_conf is None and scene_sampling_conf is None:
        raise ValueError(
            "Either scene_config_path or scene_sampling_config_path must be specified."
        )
    if scene_conf is None:
        scene_conf = {}
    if scene_sampling_conf is None:
        scene_sampling_conf = {}

    scene_result = {}

    rng = np.random.default_rng(seed=args.seed)

    light, light_position, light_energy = create_light(
        scene_conf, scene_sampling_conf, rng=rng
    )
    model = load_model(scene_conf, scene_sampling_conf)

    for i, df_scene in enumerate(
        model.sample_and_populate_scene(
            args.scene_num_samples,
            rng=rng,
        )
    ):
        objects = [obj.mesh for obj in df_scene.iloc[0]._scene.objects.values()]
        camera_poses = create_camera_poses(scene_conf, scene_sampling_conf, objects)
        data = bproc.renderer.render()

        scene_result["scm_outcomes"] = (
            df_scene.drop(columns=["_scene"]).iloc[0].to_dict()
        )
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
