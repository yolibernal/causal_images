import hashlib
import importlib
import json
import os
import shutil
import sys

import blenderproc as bproc
import dill as pickle
import numpy as np
import pandas as pd

from causal_images.scene import PrimitiveShape


# https://stackoverflow.com/a/67692
def load_module_from_file(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def save_model_component_file(
    output_dir, run_name, component_path: str, component_type: str
):
    component_name = os.path.split(component_path)[1].split(".")[0]
    with open(component_path) as f:
        file_hash = hashlib.md5(f.read().encode("utf-8")).hexdigest()
    shutil.copyfile(
        component_path,
        os.path.join(
            output_dir,
            str(run_name),
            f"{component_name}_{component_type.upper()}_{file_hash}.py",
        ),
    )


def save_outputs(
    output_dir, run_name, img_data, model, scene_result, scene_conf, scene_sampling_conf
):
    # Save rendered image
    bproc.writer.write_hdf5(os.path.join(output_dir, str(run_name)), img_data)

    # Save pickle of model
    with open(os.path.join(output_dir, str(run_name), "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open(os.path.join(output_dir, str(run_name), "scene_result.json"), "w") as f:
        json.dump(scene_result, f, cls=NumpyEncoder)

    # Save SCM plot
    fig, ax = model.plot()
    fig.savefig(os.path.join(output_dir, str(run_name), "scm.png"), dpi=fig.dpi)

    if scene_conf is not None:
        with open(
            os.path.join(output_dir, str(run_name), "scene_config.json"), "w"
        ) as f:
            json.dump(scene_conf, f, cls=NumpyEncoder)

    if scene_sampling_conf is not None:
        with open(
            os.path.join(output_dir, str(run_name), "scene_sampling_config.json"), "w"
        ) as f:
            json.dump(scene_sampling_conf, f, cls=NumpyEncoder)

        if scene_sampling_conf["scm"] is not None:
            scm_conf = scene_sampling_conf["scm"]

            if scm_conf["scm_path"] is not None:
                save_model_component_file(
                    output_dir, run_name, scm_conf["scm_path"], "scm"
                )
            if scm_conf["interventions_path"] is not None:
                save_model_component_file(
                    output_dir,
                    run_name,
                    scm_conf["interventions_path"],
                    "interventions",
                )
            if scm_conf["manipulations_path"] is not None:
                save_model_component_file(
                    output_dir,
                    run_name,
                    scm_conf["manipulations_path"],
                    "manipulations",
                )
