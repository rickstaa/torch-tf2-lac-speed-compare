"""Small test script that analysis if there is a speed difference between Pytorch and
Tensorflow when:
- Performing a forward pass through my Actor and Critic network.
"""

import timeit

# Script settings
N_SAMPLE = int(1e5)  # How many times we sample

######################################################
# Time Actor and Critic forward pass #################
######################################################
print("====Forward pass comparison Pytorch/Tensorflow====")
print(
    f"Analysing the speed of performing {N_SAMPLE} forward passes through the "
    "Actor and Critic networks..."
)

# Time Pytorch version
pytorch_setup_code = """
import torch
from gaussian_actor_torch import SquashedGaussianActor
from lyapunov_critic_torch import LyapunovCritic
# torch.set_default_tensor_type('torch.cuda.FloatTensor') # Enable global GPU
# torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
# torch.backends.cudnn.fastest = True  # Enable cudnn fastest autotuner
ga = SquashedGaussianActor(
    obs_dim=8,
    act_dim=3,
    hidden_sizes=[64, 64],
    log_std_min=-20,
    log_std_max=2,
)
lc =  LyapunovCritic(
    obs_dim=8,
    act_dim=3,
    hidden_sizes=[128, 128],
)
bs = torch.rand((256,8))
ba = torch.rand((256,3))
"""
pytorch_sample_code = """
_, _, _ = ga(bs)
_ = lc(bs, ba)
"""
print("Pytorch test...")
pytorch_time = timeit.timeit(
    pytorch_sample_code, setup=pytorch_setup_code, number=N_SAMPLE
)

# Time Tensorflow version
tf_setup_code = """
import tensorflow as tf
from gaussian_actor_tf2 import SquashedGaussianActor
from lyapunov_critic_tf2 import LyapunovCritic
# tf.config.set_visible_devices([], "GPU") # Disable GPU
ga = SquashedGaussianActor(
    obs_dim=8,
    act_dim=3,
    hidden_sizes=[64, 64],
    log_std_min=-20,
    log_std_max=2,
)
lc = LyapunovCritic(
    obs_dim=8,
    act_dim=3,
    hidden_sizes=[128, 128],
)
bs = tf.random.uniform((256,8))
ba = tf.random.uniform((256,3))
"""
tf_sample_code = """
_, _, _ = ga(bs)
_ = lc([bs, ba])
"""
print("Tensorflow test...")
tf_time = timeit.timeit(tf_sample_code, setup=tf_setup_code, number=N_SAMPLE)


######################################################
# Print results ######################################
######################################################
print("\nTest Pytorch/Tensorflow forward pass speed:")
print(f"- Pytorch forward pass time: {pytorch_time} s")
print(f"- Tf forward pass time: {tf_time} s")
