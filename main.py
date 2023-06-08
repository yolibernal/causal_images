import blenderproc as bproc

bproc.init()

import argparse
import json

from causal_images.render import render_scenes

# TODO: Add hierarchical configs
# TODO: Merge defaults
# TODO: Set BlenderProc seed (https://dlr-rm.github.io/BlenderProc/blenderproc.python.modules.main.InitializerModule.html)
# TODO: allow multiple interventions and manipulations

# Argument parsing
parser = argparse.ArgumentParser()

parser.add_argument("--output_dir", help="Output directory", required=True)

parser.add_argument(
    "--seed",
    help="Seed for initializing random generator.",
    type=int,
)

parser.add_argument(
    "--scene_num_samples",
    default=5,
    type=int,
    help="Number of sampled scenes",
)

# Scene config
parser.add_argument(
    "--scene_config_path",
)
parser.add_argument(
    "--scene_sampling_config_path",
)

args = parser.parse_args()

scene_conf = None
if args.scene_config_path is not None:
    with open(args.scene_config_path) as f:
        scene_conf = json.load(f)

scene_sampling_conf = None
if args.scene_sampling_config_path is not None:
    with open(args.scene_sampling_config_path) as f:
        scene_sampling_conf = json.load(f)

if scene_conf is None and scene_sampling_conf is None:
    raise ValueError(
        "Either scene_config_path or scene_sampling_config_path must be specified."
    )

render_scenes(args, scene_conf, scene_sampling_conf)
