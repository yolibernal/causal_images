from causal_images.scm import SceneInterventions

NOISE_SCALE3 = 6

CENTER_OFFSET = 5

interventions = SceneInterventions(
    lambda scene: {
        "pos_y3": (
            None,
            lambda noise, obj_3, pos_y1, pos_y2, weight_1, weight_2: scene.set_object_position(
                obj_3, [0, CENTER_OFFSET, noise[0] * NOISE_SCALE3]
            )[2],
            None,
        ),
    }
)
