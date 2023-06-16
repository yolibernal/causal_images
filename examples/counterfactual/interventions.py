from causal_images.scm import SceneInterventions

CENTER_OFFSET = 5
NOISE_SCALE1 = 3

INTERVENTION_OFFSET = 5

interventions = SceneInterventions(
    lambda scene: {
        "pos_1": (
            None,
            lambda noise, obj_1: scene.set_object_position(
                obj_1, [0, -CENTER_OFFSET, *noise * NOISE_SCALE1 - INTERVENTION_OFFSET]
            ),
            None,
        ),
    }
)
