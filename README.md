# Causal Images

Using [BlenderProc](https://github.com/DLR-RM/BlenderProc) to create training images from a Structural Causal Model (SCM).

## Example

### Run

Sample from the SCM and render:

`blenderproc run main.py sample --output_dir outputs/basic --scene_num_samples=5 --sampling_config examples/basic/sampling.json --fixed_config examples/basic/config.json`

Generate counterfactuals from previous output:

`blenderproc run main.py counterfactual --input_dir outputs/basic  --output_dir outputs/basic_counterfactual  --interventions_path examples/counterfactual/interventions.py`

Export output as data/labels:

`blenderproc run main.py export --input_dir outputs/basic --output_dir outputs/basic_export --to-image`

### Visualize

Visualize a generated image:

`blenderproc vis hdf5 outputs/basic/0/0.hdf5`
