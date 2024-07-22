import numpy as np
from sympy.stats import Beta, Uniform

from causal_images.scene import PrimitiveShape
from causal_images.scm import SceneSCM

NOISE_SCALE1 = 4
NOISE_SCALE2 = 4
NOISE_SCALE3 = 4

# TODO: do not scale in z-axis
# TODO: intervene weights / set constant weights

WEIGHT1 = 1.0
WEIGHT2 = 0.7
WEIGHT3 = 0.5

scm = SceneSCM(
    lambda scene: {
        # "weight_1": (
        #     [],
        #     lambda noise: noise[0],
        #     Beta("P", 3, 1),
        # ),
        "weight_1": (
            [],
            lambda: WEIGHT1,
            None,
        ),
        "obj_1": (
            ["weight_1"],
            lambda weight_1: scene.create_primitive(
                PrimitiveShape.CUBE,
                material_name="Red",
                # scale=np.repeat(weight_1, 3),
            ),
            None,
        ),
        "pos_x1": (
            ["obj_1"],
            lambda noise, obj_1: scene.set_object_position(
                obj_1, [noise[0] * NOISE_SCALE1, None, 0]
            )[0],
            Uniform("P", -1, 1),
        ),
        "pos_y1": (
            ["obj_1"],
            lambda noise, obj_1: scene.set_object_position(
                obj_1, [None, noise[0] * NOISE_SCALE1, 0]
            )[1],
            Uniform("P", -1, 1),
        ),
        # "weight_2": (
        #     [],
        #     lambda noise: noise[0],
        #     Beta("P", 2, 1),
        # ),
        "weight_2": (
            [],
            lambda: WEIGHT2,
            None,
        ),
        "obj_2": (
            ["weight_2"],
            lambda weight_2: scene.create_primitive(
                PrimitiveShape.CUBE,
                material_name="Green",
                # scale=np.repeat(weight_2, 3),
            ),
            None,
        ),
        "pos_x2": (
            ["obj_2"],
            lambda noise, obj_2: scene.set_object_position(
                obj_2, [noise[0] * NOISE_SCALE2, None, 0]
            )[0],
            Uniform("P", -1, 1),
        ),
        "pos_y2": (
            ["obj_2"],
            lambda noise, obj_2: scene.set_object_position(
                obj_2, [None, noise[0] * NOISE_SCALE2, 0]
            )[1],
            Uniform("P", -1, 1),
        ),
        "obj_3": (
            [],
            lambda: scene.create_primitive(
                PrimitiveShape.CUBE,
                material_name="Blue",
                # scale=np.repeat(WEIGHT3, 3),
            ),
            None,
        ),
        "pos_x3": (
            ["obj_3", "pos_x1", "pos_x2", "weight_1", "weight_2"],
            lambda noise, obj_3, pos_x1, pos_x2, weight_1, weight_2: scene.set_object_position(
                obj_3,
                [
                    ((weight_1 * pos_x1 + weight_2 * pos_x2) / (weight_1 + weight_2))
                    + (noise[0] * NOISE_SCALE3),
                    None,
                    0,
                ],
            )[0],
            Uniform("P", -1, 1),
        ),
        "pos_y3": (
            ["obj_3", "pos_y1", "pos_y2", "weight_1", "weight_2"],
            lambda noise, obj_3, pos_y1, pos_y2, weight_1, weight_2: scene.set_object_position(
                obj_3,
                [
                    None,
                    ((weight_1 * pos_y1 + weight_2 * pos_y2) / (weight_1 + weight_2))
                    + (noise[0] * NOISE_SCALE3),
                    0,
                ],
            )[1],
            Uniform("P", -1, 1),
        ),
    }
)
