import hashlib
import importlib
import json
import os
import shutil
import sys

import blenderproc as bproc
import bpy
import dill as pickle
import h5py
import numpy as np
from PIL import Image


# https://stackoverflow.com/a/67692
def load_module_from_file(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def save_model_component_file(output_dir, component_path: str, component_type: str):
    component_name = os.path.split(component_path)[1].split(".")[0]
    with open(component_path) as f:
        file_hash = hashlib.md5(f.read().encode("utf-8")).hexdigest()
    shutil.copyfile(
        component_path,
        os.path.join(
            output_dir,
            f"{component_name}_{component_type.upper()}_{file_hash}.py",
        ),
    )


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_run_config(output_dir, model, fixed_conf, sampling_conf):
    # Save pickle of model
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    if fixed_conf is not None:
        with open(os.path.join(output_dir, "fixed_config.json"), "w") as f:
            json.dump(fixed_conf, f, cls=NumpyEncoder)

    if sampling_conf is not None:
        # Save scene sampling config
        with open(os.path.join(output_dir, "sampling_config.json"), "w") as f:
            json.dump(sampling_conf, f, cls=NumpyEncoder)

        # Save referenced files
        if "scm" in sampling_conf:
            scm_conf = sampling_conf["scm"]

            if scm_conf["scm_path"] is not None:
                save_model_component_file(output_dir, scm_conf["scm_path"], "scm")
            if scm_conf["interventions_path"] is not None:
                save_model_component_file(
                    output_dir,
                    scm_conf["interventions_path"],
                    "interventions",
                )
            if scm_conf["manipulations_path"] is not None:
                save_model_component_file(
                    output_dir,
                    scm_conf["manipulations_path"],
                    "manipulations",
                )

        # Save SCM plot
        # fig, ax = model.plot()
        # fig.savefig(os.path.join(output_dir, "scm.png"), dpi=fig.dpi)


def save_run_outputs(
    output_dir,
    run_name,
    img_data,
    scene_result,
):
    run_dir = os.path.join(output_dir, str(run_name))

    os.makedirs(run_dir, exist_ok=True)

    # Save rendered image
    if img_data is not None:
        bproc.writer.write_hdf5(run_dir, img_data)

    with open(os.path.join(run_dir, "scene_result.json"), "w") as f:
        json.dump(scene_result, f, cls=NumpyEncoder)


def hdf5_to_image(input_path):
    hdf = h5py.File(input_path, "r")
    colors = np.array(hdf["colors"])
    color_mode = bpy.context.scene.render.image_settings.color_mode
    img = Image.fromarray(colors.astype("uint8"), color_mode)
    return img
