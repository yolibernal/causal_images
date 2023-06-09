# Causal Images

Using [BlenderProc](https://github.com/DLR-RM/BlenderProc) to create training images from a Structural Causal Model (SCM).

## Example

### Run

Sample from the SCM and render:
`blenderproc run main.py --output_dir <OUTPUT_DIR> --sampling_config sampling.json --fixed_config fixed.json`

### Visualize

Visualize a generated image:

`blenderproc vis hdf5 output/0/0.hdf5`
