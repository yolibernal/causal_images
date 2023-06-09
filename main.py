import blenderproc as bproc

bproc.init()

import argparse
import json
import os

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
    "--fixed_config",
)
parser.add_argument(
    "--sampling_config",
)

args = parser.parse_args()

fixed_conf = None
if args.fixed_config is not None:
    with open(args.fixed_config) as f:
        fixed_conf = json.load(f)

sampling_conf = None
if args.sampling_config is not None:
    with open(args.sampling_config) as f:
        sampling_conf = json.load(f)

if fixed_conf is None and sampling_conf is None:
    raise ValueError("Either fixed_config or sampling_config must be specified.")

if args.seed is not None:
    os.environ["BLENDER_PROC_RANDOM_SEED"] = args.seed

render_scenes(args, fixed_conf, sampling_conf)
