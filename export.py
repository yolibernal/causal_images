import argparse
import os
import shutil

import h5py
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", help="Input directory", required=True)
parser.add_argument("--output_dir", help="Output directory", required=True)

parser.add_argument("--to-image", help="Convert hdf5 to image", action="store_true")

args = parser.parse_args()

output_dir = args.output_dir
input_dir = args.input_dir
to_image = args.to_image

file_index = 0

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def hdf5_to_image(input_path, output_path, format="JPEG"):
    hdf = h5py.File(input_path, "r")
    colors = np.array(hdf["colors"])
    img = Image.fromarray(colors.astype("uint8"), "RGB")
    img.save(output_path, format)


for subdir, dirs, files in os.walk(input_dir):
    for file in files:
        filepath = os.path.join(subdir, file)
        if filepath.endswith(".hdf5"):
            if to_image:
                hdf5_to_image(filepath, os.path.join(output_dir, f"{file_index}.jpg"))
            else:
                shutil.copy(filepath, os.path.join(output_dir, f"{file_index}.hdf5"))
            file_index += 1
