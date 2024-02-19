## Seq-GFN-SR: a refactored library for sequential equation generation with GFlowNets

This repo is adapted from the vanilla version implemented at [here](https://github.com/listar2000/gfn-sr/tree/draft). It leverages the `TorchGFN` 
[framework](https://github.com/GFNOrg/torchgfn) developed at MILA.

### 1. Get Started: Generate Synthetic Data

For any symbolic regression (SR) task, you need to provide a covariates matrix `X` of shape `(n, d)`
and a response array `y` of shape `(n,)`. A handy way to generate such dataset is through the `utils/data_utils.py`.

For example, to obtain 20 samples from the function `sin(x) + y`, you can simply call:
```bash
python utils/data_utils.py --exp "x+sin(z)" -n 20 --ranges x -1 1 --ranges z 0 1
```
which provides a string-based approach to create synthetic data (via `sympy`). The generated data will be stored
in a file `data.npz` by default. You can specify the file info by supplying `--save` (or `-s`) command.

### 2. Set up the configuration file
Almost all of hyperparameters in Seq-GFN-SR is specified through a config json file (an example and default one is available at `config.json`).
Remember to replace the `data` file location with the actual one used in the first step.

### 3. Run training script
You should be all set at this moment and simply run `python train.py --config config.json` is all you need.