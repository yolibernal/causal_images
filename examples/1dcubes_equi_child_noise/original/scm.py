from sympy.stats import Beta, Uniform

from causal_images.scene import PrimitiveShape
from causal_images.scm import SceneSCM

CENTER_OFFSET = 5

NOISE_SCALE1 = 4.0
NOISE_SCALE2 = 4.0
NOISE_SCALE3 = 4.0

WEIGHT1 = 1
WEIGHT2 = 0.7

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
                # material_name="Red",
                # scale=np.repeat(weight_1, 3),
            ),
            None,
        ),
        "pos_y1": (
            ["obj_1"],
            lambda noise, obj_1: scene.set_object_position(
                obj_1, [0, -CENTER_OFFSET, noise[0] * NOISE_SCALE1]
            )[2],
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
                # material_name="Green",
                # scale=np.repeat(weight_2, 3),
            ),
            None,
        ),
        "pos_y2": (
            ["obj_2"],
            lambda noise, obj_2: scene.set_object_position(obj_2, [0, 0, noise[0] * NOISE_SCALE2])[
                2
            ],
            Uniform("P", -1, 1),
        ),
        "obj_3": (
            [],
            lambda: scene.create_primitive(
                PrimitiveShape.CUBE,
                # material_name="Blue",
                # scale=np.repeat(WEIGHT3, 3),
            ),
            None,
        ),
        "pos_y3": (
            ["obj_3", "pos_y1", "pos_y2", "weight_1", "weight_2"],
            lambda noise, obj_3, pos_y1, pos_y2, weight_1, weight_2: scene.set_object_position(
                obj_3,
                [
                    0,
                    CENTER_OFFSET,
                    ((weight_1 * pos_y1 + weight_2 * pos_y2) / (weight_1 + weight_2))
                    + (noise[0] * NOISE_SCALE3),
                ],
            )[2],
            Uniform("P", -1, 1),
        ),
    }
)
