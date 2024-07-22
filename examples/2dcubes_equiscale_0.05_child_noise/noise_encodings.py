NOISE_SCALE = 6
CHILD_NOISE_SCALE = 0.3


def noise_encodings_from_causal_variables(
    pos_x1,
    pos_y1,
    pos_x2,
    pos_y2,
    pos_x3,
    pos_y3,
    weight_1,
    weight_2,
    noise_scale=NOISE_SCALE,
    noise_scale_children=CHILD_NOISE_SCALE,
):
    e_pos_x1 = pos_x1 / noise_scale
    e_pos_y1 = pos_y1 / noise_scale
    e_pos_x2 = pos_x2 / noise_scale
    e_pos_y2 = pos_y2 / noise_scale

    mean_position_x = (pos_x1 * weight_1 + pos_x2 * weight_2) / (weight_1 + weight_2)
    mean_position_y = (pos_y1 * weight_1 + pos_y2 * weight_2) / (weight_1 + weight_2)
    e_pos_x3 = (pos_x3 - mean_position_x) / noise_scale_children
    e_pos_y3 = (pos_y3 - mean_position_y) / noise_scale_children

    noise_encodings = {
        "pos_x1": e_pos_x1,
        "pos_y1": e_pos_y1,
        "pos_x2": e_pos_x2,
        "pos_y2": e_pos_y2,
        "pos_x3": e_pos_x3,
        "pos_y3": e_pos_y3,
    }

    return noise_encodings


computed_values = {
    "noise_encodings": lambda scene_result: noise_encodings_from_causal_variables(
        scene_result["scm_outcomes"]["pos_x1"],
        scene_result["scm_outcomes"]["pos_y1"],
        scene_result["scm_outcomes"]["pos_x2"],
        scene_result["scm_outcomes"]["pos_y2"],
        scene_result["scm_outcomes"]["pos_x3"],
        scene_result["scm_outcomes"]["pos_y3"],
        scene_result["scm_outcomes"]["weight_1"],
        scene_result["scm_outcomes"]["weight_2"],
        noise_scale=NOISE_SCALE,
        noise_scale_children=CHILD_NOISE_SCALE,
    )
}
