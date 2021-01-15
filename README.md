# Torch TF2 bijection speed compare

This repository was created to investigate a speed difference, that was found between
Pytorch and Tensorflow 2.x (Eager mode), when training a Lyapunov Critic RL algorithm.
It currently contains the following comparison scripts:

-   [timeit_forward_full_pass_speed_compare.py](https://github.com/rickstaa/torch_tf2_lac_speed_compare/blob/main/timeit_forward_full_pass_speed_compare.py):
      Script used to compare the speed of a forward pass through the networks.
-   [timeit_lac_algorithm_speed_compare.py](https://github.com/rickstaa/torch_tf2_lac_speed_compare/blob/main/timeit_lac_algorithm_speed_compare.py): Script used to compare the speed of training the full LAC algorithm.
-   [timeit_log_prob_speed_compare.py](https://github.com/rickstaa/torch_tf2_lac_speed_compare/blob/main/timeit_log_prob_speed_compare.py): Script used to compare the speed of calculating the log_probability of a normal distribution.
-   [timeit_log_prob_squash_speed_compare.py](https://github.com/rickstaa/torch_tf2_lac_speed_compare/blob/main/timeit_log_prob_squash_speed_compare.py): Script used to compare the speed of calculating the log_probability of a normal distribution and performing a squashing correction on this distribution.
-   [timeit_rsample_log_prob_speed_compare.py](https://github.com/rickstaa/torch_tf2_lac_speed_compare/blob/main/timeit_rsample_log_prob_speed_compare.py): Script used to compare the speed of rsampling from a normal distribution while also calculating the log_probability of this sampled action.
-   [timeit_rsample_log_prob_squash_speed_compare.py](https://github.com/rickstaa/torch_tf2_lac_speed_compare/blob/main/timeit_rsample_log_prob_squash_speed_compare.py): Script used to compare the speed of rsampling from a normal distribution, calculating the log probability and performing a squashing correction.
-   [timeit_sample_speed_compare.py](https://github.com/rickstaa/torch_tf2_lac_speed_compare/blob/main/timeit_sample_speed_compare.py): Script used to compare the speed of sampling/rsampling from a normal distribution.

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

To see a speed comparison run the following command:

```bash
python <SCRIPT_NAME>.py
```

#### GPU settings

In each script you can disable or enable the GPU. For the tensorflow version you can disable the GPU by un-commenting the following line:

```python
tf.config.set_visible_devices([], "GPU") # Disable GPU`
```

In PYtorch you can enable the gpu by un-commenting the following lines:

```python
# torch.set_default_tensor_type('torch.cuda.FloatTensor') # Enable global GPU
# torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
# torch.backends.cudnn.fastest = True  # Enable cudnn fastest autotuner
```

The last two modes are related to the settings for the cudnn autotuner (see [this forum post](https://duckduckgo.com/?q=torch.backends.cudnn.fastest&t=newext&atb=v243-1&ia=web) for more information).
