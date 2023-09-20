import numpy as np
from sympy.stats import Uniform

from causal_images.scene import PrimitiveShape
from causal_images.scm import SceneSCM

CENTER_OFFSET = 5
NOISE_SCALE1 = 3
NOISE_SCALE2 = 3

scm = SceneSCM(
    lambda scene: {
        "obj_1": ([], lambda: scene.create_primitive(PrimitiveShape.CUBE), None),
        "pos_1": (
            ["obj_1"],
            lambda noise, obj_1: scene.set_object_position(
                obj_1, [0, -CENTER_OFFSET, *noise * NOISE_SCALE1]
            ),
            Uniform("P", -1, 1),
        ),
        "obj_2": ([], lambda: scene.create_primitive(PrimitiveShape.CUBE), None),
        "pos_2": (
            ["obj_2"],
            lambda noise, obj_2: scene.set_object_position(obj_2, [0, 0, *noise * NOISE_SCALE2]),
            Uniform("P", -1, 1),
        ),
        "obj_3": ([], lambda: scene.create_primitive(PrimitiveShape.CUBE), None),
        "pos_3": (
            ["obj_3", "pos_1", "pos_2"],
            lambda obj_3, pos_1, pos_2: scene.set_object_position(
                obj_3, [0, CENTER_OFFSET, pos_1[2] + pos_2[2]]
            ),
            None,
        ),
    }
)
