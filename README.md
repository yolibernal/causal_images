# Causal Images

Using [BlenderProc](https://github.com/DLR-RM/BlenderProc) to create training images from a Structural Causal Model (SCM).

## Example

### Run

Sample from the SCM and render:

`blenderproc run sample.py --output_dir outputs/basic --scene_num_samples=5 --sampling_config examples/basic/sampling.json --fixed_config examples/basic/config.json`

Generate counterfactuals from previous output:

`blenderproc run counterfactual.py --input_dir outputs/basic  --output_dir outputs/basic_counterfactual  --interventions examples/counterfactual/interventions.py`

### Visualize

Visualize a generated image:

`blenderproc vis hdf5 outputs/basic/0/0.hdf5`
