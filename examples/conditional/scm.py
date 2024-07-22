from sympy.stats import Uniform

from causal_images.scene import PrimitiveShape, RelativePosition
from causal_images.scm import SceneSCM

scm = SceneSCM(
    lambda scene: {
        "obj_1": (
            [],
            lambda noise: (
                scene.create_primitive(
                    PrimitiveShape.CUBE if noise[0] < 0 else PrimitiveShape.SPHERE,
                    material_name="Blue",
                )
            ),
            Uniform("P", -1, 1),
        ),
        "pos_1": (
            ["obj_1"],
            lambda noise, obj_1: scene.set_object_position(obj_1, [0, noise[0], 0]),
            Uniform("Q", -1, 1),
        ),
        "obj_2": (
            ["obj_1"],
            lambda obj_1: (
                scene.create_primitive(
                    (
                        PrimitiveShape.CONE
                        if scene.objects[obj_1].shape == PrimitiveShape.CUBE
                        else PrimitiveShape.CYLINDER
                    ),
                    material_name="Red",
                )
            ),
            None,
        ),
        "pos_2": (
            ["obj_1", "obj_2", "pos_1"],
            lambda obj_1, obj_2, pos_1: scene.set_object_relative_position(
                obj_2,
                obj_1,
                pos_1,
                (RelativePosition.TOP if obj_2 == PrimitiveShape.CONE else RelativePosition.BOTTOM),
            ),
            None,
        ),
    }
)
