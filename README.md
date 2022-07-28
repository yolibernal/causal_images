# Causal Images

Using [BlenderProc](https://github.com/DLR-RM/BlenderProc) to create training images from a Structural Causal Model (SCM).

## Example

### Run

Sample from the SCM and render:

`blenderproc run main.py --output_dir ./output --scm_path ./examples/scm.py --scene_num_samples 1 --camera_num_samples 1 --camera_azimuth_lower 50 --camera_azimuth_upper 180 --camera_elevation_lower 20 --camera_elevation_upper 60 --light_position 0 0 10 --light_energy 5000`

### Visualize

Visualize a generated image:

`blenderproc vis hdf5 output/0/0.hdf5`
