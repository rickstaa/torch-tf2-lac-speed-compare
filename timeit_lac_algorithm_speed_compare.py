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
print("====Full learn method comparison Pytorch/Tensorflow====")
print(f"Analysing the speed of training a LAC algorithm using {N_SAMPLE} episodes.")

# Time Pytorch version
pytorch_setup_code = """
import torch

from lac_trimmed_torch import LAC

# torch.set_default_tensor_type('torch.cuda.FloatTensor') # Enable global GPU
# torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
# torch.backends.cudnn.fastest = True  # Enable cudnn fastest autotuner

# Create dummy batch
batch_size=256
act_dim = 3
obs_dim = 8
batch = {
    "s": torch.rand((batch_size, obs_dim)),
    "a": torch.rand((batch_size, act_dim)),
    "r": torch.rand((batch_size, 1)),
    "terminal": torch.randint(0, 2, (batch_size,1)),
    "s_": torch.rand((batch_size, obs_dim)),
}

# Put batch on GPU if requested

# Create LAC
LAC = LAC()
"""
pytorch_sample_code = """
LAC.learn(batch)
"""
print("Pytorch test...")
pytorch_time = timeit.timeit(
    pytorch_sample_code, setup=pytorch_setup_code, number=N_SAMPLE
)

# Time Tensorflow version
tf_setup_code = """
import tensorflow as tf

from lac_trimmed_tf2 import LAC

# tf.config.set_visible_devices([], "GPU") # Disable GPU

# Create dummy batch
batch_size=256
act_dim = 3
obs_dim = 8
batch = {
    "s": tf.random.uniform((batch_size, obs_dim)),
    "a": tf.random.uniform((batch_size, act_dim)),
    "r": tf.random.uniform((batch_size, 1)),
    "terminal": tf.cast(
        tf.random.uniform((batch_size, 1), minval=0, maxval=2, dtype=tf.int32),
        dtype=tf.float32,
    ),
    "s_": tf.random.uniform((batch_size, obs_dim)),
}

# Create LAC
LAC = LAC()
"""
tf_sample_code = """
LAC.learn(batch)
"""
print("Tensorflow test...")
tf_time = timeit.timeit(tf_sample_code, setup=tf_setup_code, number=N_SAMPLE)


######################################################
# Print results ######################################
######################################################
print("\nTest Pytorch/Tensorflow lac algorithm speed:")
print(f"- Pytorch lac algorithm time: {pytorch_time} s")
print(f"- Tf lac algorithm time: {tf_time} s")
