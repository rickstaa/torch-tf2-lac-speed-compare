# Torch TF2 bijection speed compare

This repository was created to investigate a speed difference, that was found between
Pytorch and Tensorflow 2.x (Eager mode), when performing a squashing operation onto a
normal distribution

## Use instructions

### Conda environment

From the general python package sanity perspective, it is a good idea to use conda
environments to make sure packages from different projects do not interfere with
each other.

To create a conda env with python3, one runs

```bash
conda create -n torch_tf2_speed_compare python=3.8
```

To activate the env:

```bash
conda activate torch_tf2_speed_compare
```

### Installation Environment

```bash
pip install -r requirements.txt
```

Then you are free to run main.py to train agents. Hyperparameters for training LAC in Cartpole are ready to run by default. If you would like to test other environments and algorithms, please open variant.py and choose corresponding 'env_name' and 'algorithm_name'.

### Usage instructions
