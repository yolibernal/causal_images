from typing import List

import blenderproc as bproc
import numpy as np


# https://b3d.interplanety.org/en/how-to-calculate-the-bounding-sphere-for-selected-objects/
def get_bounding_sphere(objects: List[bproc.types.MeshObject]):
    # return the bounding sphere center and radius for objects (in global coordinates)
    if not isinstance(objects, list):
        objects = [objects]
    points_co_global = []

    for obj in objects:
        points_co_global.extend(obj.get_bound_box())

    def get_center(l):
        return (max(l) + min(l)) / 2 if l else 0.0

    x, y, z = [[point_co[i] for point_co in points_co_global] for i in range(3)]
    b_sphere_center = (
        np.array([get_center(axis) for axis in [x, y, z]]) if (x and y and z) else None
    )
    b_sphere_radius_vector = (
        np.amax([(point - b_sphere_center) for point in points_co_global], axis=0)
        if b_sphere_center is not None
        else None
    )

    return b_sphere_center, np.linalg.norm(b_sphere_radius_vector)


def sample_object_facing_camera_pose(
    objects: List[
        bproc.types.MeshObject,
    ],
    fov_bounds=(np.pi / 6, np.pi / 6),
    camera_zoom_bounds=(1, 1),
    camera_rotation_bounds=(0, 0),
    camera_elevation_bounds=(-1, 1),
    camera_azimuth_bounds=(-1, 1),
):
    bounding_sphere_center, bounding_sphere_radius = get_bounding_sphere(objects)

    r = bounding_sphere_radius
    fov = np.random.uniform(fov_bounds[0], fov_bounds[1])

    # Distance to make bounding sphere fit into fov
    d = r / np.sin(fov / 2)

    bproc.camera.set_intrinsics_from_blender_params(
        lens=fov, lens_unit="FOV", clip_end=d + r
    )

    # Sample location
    location = bproc.sampler.shell(
        center=bounding_sphere_center,
        radius_min=d * camera_zoom_bounds[0],
        radius_max=d * camera_zoom_bounds[1],
        elevation_min=camera_elevation_bounds[0],
        elevation_max=camera_elevation_bounds[1],
        azimuth_min=camera_azimuth_bounds[0],
        azimuth_max=camera_azimuth_bounds[1],
    )
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        bounding_sphere_center - location,
        inplane_rot=np.random.uniform(
            camera_rotation_bounds[0] * np.pi, camera_rotation_bounds[1] * np.pi
        ),
    )
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    return cam2world_matrix
