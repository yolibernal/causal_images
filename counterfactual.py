import blenderproc as bproc

bproc.init()

import argparse
import json
import os

from causal_images.render import render_scenes_from_configs

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", help="Input directory", required=True)
parser.add_argument("--output_dir", help="Output directory", required=True)

parser.add_argument("--interventions", help="Interventions file", required=True)

parser.add_argument(
    "--seed",
    help="Seed for initializing random generator.",
    type=int,
    default=0,
)

args = parser.parse_args()

# input_dir = "./outputs/3node"
input_dir = args.input_dir
# output_dir = "./outputs/3node_counterfactual"
output_dir = args.output_dir

interventions_path = args.interventions

# interventions_path = "./configs/3node_counterfactual/interventions.py"

seed = args.seed

# Load the scm
scm_paths = [
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if "_SCM_" in f and f.endswith(".py")
]
if len(scm_paths) != 1:
    raise ValueError("Expected exactly one SCM file.")
scm_path = scm_paths[0]
# scm_module = load_module_from_file(scm_path, "scm")
# model = scm_module.scm

sampling_conf = {
    "scm": {
        "scm_path": scm_path,
        "interventions_path": interventions_path,
        "manipulations_path": None,
    }
}
excluded_dirs = ["__pycache__"]
run_directories = next(os.walk(input_dir))[1]
for run_dir in run_directories:
    if run_dir in excluded_dirs:
        continue
    scene_result_path = os.path.join(input_dir, run_dir, "scene_result.json")
    if not os.path.exists(scene_result_path):
        raise ValueError(f"Scene result not found for {run_dir}.")
    with open(scene_result_path) as f:
        scene_result = json.load(f)
    fixed_conf = {k: v for k, v in scene_result.items() if k not in ["scm_outcomes"]}
    render_scenes_from_configs(
        fixed_conf=fixed_conf,
        sampling_conf=sampling_conf,
        seed=seed,
        scene_num_samples=1,
        output_dir=output_dir,
        run_names=[run_dir],
    )
