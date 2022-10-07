import numpy as np
from causal_images.scene import PrimitiveShape, RelativePosition
from causal_images.scm import SceneSCM

scm = SceneSCM(
    lambda scene, rng: {
        "obj_x": ([], lambda: scene.create_primitive(PrimitiveShape.CUBE), None),
        "pos_x": (
            ["obj_x"],
            lambda obj_x: scene.set_object_position(
                obj_x, rng.normal(loc=0, scale=1, size=(3,))
            ),
            None,
        ),
        "obj_y": ([], lambda: scene.create_primitive(PrimitiveShape.SPHERE), None),
        "pos_y": (
            ["obj_y"],
            lambda obj_y: scene.set_object_position(
                obj_y, rng.normal(loc=5, scale=1, size=(3,))
            ),
            None,
        ),
        "obj_z": ([], lambda: scene.create_primitive(PrimitiveShape.CONE), None),
        "pos_z": (
            ["obj_z", "obj_y", "pos_y"],
            lambda obj_z, obj_y, pos_y: scene.set_object_relative_position(
                obj_z, obj_y, pos_y, RelativePosition.TOP
            ),
            None,
        ),
    }
)
