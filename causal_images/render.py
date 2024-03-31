import json

import blenderproc as bproc
import bpy
import numpy as np
from blenderproc.python.utility.CollisionUtility import CollisionUtility

from causal_images.camera import sample_object_facing_camera_pose
from causal_images.scm import SceneInterventions, SceneManipulations, SceneSCM
from causal_images.util import load_module_from_file, save_run_config, save_run_outputs


def create_light_from_config(fixed_conf, sampling_conf):
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
        if "poses" in fixed_conf["camera"]:
            camera_poses = np.array(fixed_conf["camera"]["poses"])

        if "type" in fixed_conf["camera"]:
            cam_ob = bpy.context.scene.camera
            cam_ob.data.type = fixed_conf["camera"]["type"]
            if cam_ob.data.type == "ORTHO":
                cam_ob.data.ortho_scale = fixed_conf["camera"].get("ortho_scale", 5)
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
    fixed_noise_values = None
    if "scm_noise_values" in fixed_conf:
        fixed_noise_values = fixed_conf["scm_noise_values"]

    # Render scenes from fixed SCM outcomes
    if "scm_outcomes" in fixed_conf:
        model = SceneSCM.from_scm_outcomes(fixed_conf["scm_outcomes"])
    # Sample scenes from SCM
    elif "scm" in sampling_conf:
        scm_conf = sampling_conf["scm"]

        scm = load_module_from_file(scm_conf["scm_path"], "scm")
        model: SceneSCM = scm.scm

        if scm_conf["interventions_path"] is not None:
            interventions = load_module_from_file(scm_conf["interventions_path"], "interventions")
            model_interventions: SceneInterventions = interventions.interventions
            model.interventions = model_interventions

        if scm_conf["manipulations_path"] is not None:
            manipulations = load_module_from_file(scm_conf["manipulations_path"], "manipulations")
            model_manipulations: SceneManipulations = manipulations.manipulations
            model.manipulations = model_manipulations
        if fixed_noise_values is not None:
            model.fixed_noise_values = fixed_noise_values
    else:
        raise ValueError("SCM config not specified.")
    return model


def load_computed_values(fixed_conf):
    computed_values = None
    if "computed_values_path" in fixed_conf:
        # load module
        computed_values = load_module_from_file(
            fixed_conf["computed_values_path"], "computed_values"
        )
        computed_values: dict[str, callable] = computed_values.computed_values
    return computed_values


def load_material_library(material_library_path):
    regex = r"^(?!Dots Stroke).*$"
    objs = bproc.loader.load_blend(
        material_library_path, name_regrex=regex, data_blocks="materials"
    )
    materials = bproc.material.collect_all()
    return materials


def check_for_collision(objects):
    for first_obj_index, first_obj in enumerate(objects):
        for second_obj in objects[first_obj_index + 1 :]:
            objects_collide, _ = CollisionUtility.check_mesh_intersection(first_obj, second_obj)
            if objects_collide == True:
                return True
    return False


def render_scenes_from_configs(
    fixed_conf,
    sampling_conf,
    material_library_path,
    seed,
    scene_num_samples,
    output_dir,
    run_names=None,
    allow_collisions=False,
    enable_transparency=True,
    skip_render=False,
):
    if fixed_conf is None and sampling_conf is None:
        raise ValueError("Either fixed_conf or sampling_conf must be specified.")
    if fixed_conf is None:
        fixed_conf = {}
    if sampling_conf is None:
        sampling_conf = {}

    scene_result = {}

    rng = np.random.default_rng(seed=seed)

    resolution = fixed_conf.get("resolution", [512, 512])
    bproc.camera.set_resolution(*resolution)
    scene_result["resolution"] = resolution

    light, light_position, light_energy = create_light_from_config(fixed_conf, sampling_conf)

    model = load_model(fixed_conf, sampling_conf)
    computed_values = load_computed_values(fixed_conf)

    load_materials = material_library_path is not None and material_library_path != ""
    materials = None
    if load_materials:
        materials = load_material_library(material_library_path)

    num_rendered_scenes = 0

    scene_generator = model.sample_and_populate_scene(n=-1, rng=rng)

    bproc.renderer.set_output_format(enable_transparency=enable_transparency)

    # bproc.world.set_world_background_hdr_img("alps_field_1k.hdr")

    while num_rendered_scenes < scene_num_samples:
        # if num_rendered_scenes % clean_every_n_scenes == 0:
        #     bproc.clean_up()
        #     if load_materials:
        #         materials = load_material_library(material_library_path)
        i = num_rendered_scenes
        scm_outcomes, scm_noise_values, scene = next(scene_generator)

        objects = [obj.mesh for obj in scene.objects.values()]
        camera_poses = create_camera_poses(fixed_conf, sampling_conf, objects)

        if not allow_collisions:
            collision_detected = check_for_collision(objects)
            if collision_detected:
                print("Collision detected. Skipping scene.")
                continue

        scene_result["scm_outcomes"] = scm_outcomes
        scene_result["scm_noise_values"] = scm_noise_values
        scene_result["camera"] = {
            **fixed_conf["camera"],
            "poses": camera_poses.tolist(),
        }
        scene_result["light"] = {
            "position": light_position,
            "energy": light_energy,
        }

        if computed_values is not None:
            scene_result.update(
                {key: value(scene_result) for key, value in computed_values.items()}
            )

        if skip_render:
            data = None
            # Cleaning scene leads to crash due to Blender bug (executing unreachable Blender code)
            # scene.cleanup()
        else:
            data = bproc.renderer.render()

        save_run_outputs(
            output_dir=output_dir,
            run_name=run_names[i] if run_names is not None else i,
            img_data=data,
            scene_result=scene_result,
        )

        num_rendered_scenes += 1

    # Clean up last scene (TODO: Use context manager)
    scene.cleanup()

    save_run_config(
        output_dir=output_dir,
        model=model,
        fixed_conf=fixed_conf,
        sampling_conf=sampling_conf,
    )
    light.delete()
