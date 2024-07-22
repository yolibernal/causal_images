from causal_images.scm import SceneInterventions

NOISE_SCALE3 = 6

interventions = SceneInterventions(
    lambda scene: {
        "pos_x3": (
            None,
            lambda noise, obj_3, pos_x1, pos_x2, weight_1, weight_2: scene.set_object_position(
                obj_3, [0, noise[0] * NOISE_SCALE3, None]
            )[1],
            None,
        ),
    }
)
