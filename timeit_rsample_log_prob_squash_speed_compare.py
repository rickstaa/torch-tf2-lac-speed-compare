"""Small test script that analysis if there is a speed difference between Pytorch and
Tensorflow when:
- R sampling a action from a normal distribution
- Calculating the Log probability based on this sample.
- Squashing the action.
- Normal distribution squash correction (_forward_log_det_jacobian).
"""
import timeit

# Script settings
N_SAMPLE = int(1e5)  # How many times we sample

######################################################
# Time rsample + logprob calculation + squash ########
######################################################
print("====rsample + log_prob + squash speed comparison Pytorch/Tensorflow====")
print(
    "Analysing the speed of calculating the log_prob of an action that is "
    f"r-sampled from the normal distribution {N_SAMPLE} times. While also squashing "
    "the output..."
)

# Time Pytorch version
pytorch_setup_code = """
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
# torch.set_default_tensor_type('torch.cuda.FloatTensor') # Enable global GPU
# torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
# torch.backends.cudnn.fastest = True  # Enable cudnn fastest autotuner
batch_size=256
mu = torch.zeros(batch_size, 3)
std = torch.ones(batch_size, 3)
"""
pytorch_sample_code = """
pi_distribution = Normal(torch.zeros(batch_size, 3), torch.ones(batch_size, 3))
pi_action = (
    pi_distribution.rsample()
)  # Sample while using the parameterization trick
pi_action = torch.tanh(pi_action)  # Squash gaussian to be between -1 and 1
logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
    axis=1
)
"""
print("Pytorch test...")
pytorch_time = timeit.timeit(
    pytorch_sample_code, setup=pytorch_setup_code, number=N_SAMPLE
)

# Time Tensorflow version
tf_setup_code = """
import tensorflow as tf
import tensorflow_probability as tfp
from squash_bijector import SquashBijector
# tf.config.set_visible_devices([], "GPU") # Disable GPU
batch_size=256
mu = tf.zeros((batch_size, 3), dtype=tf.float32)
std = tf.ones((batch_size, 3), dtype=tf.float32)
@tf.function
def sample_function():
    squash_bijector = SquashBijector()
    affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(std))
    base_distribution = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(3), scale_diag=tf.ones(3)
    )
    epsilon = base_distribution.sample(batch_size)
    raw_action = affine_bijector.forward(epsilon)
    pi_action = squash_bijector.forward(raw_action)
    reparm_trick_bijector = tfp.bijectors.Chain((squash_bijector, affine_bijector))
    distribution = tfp.distributions.TransformedDistribution(
        distribution=base_distribution, bijector=reparm_trick_bijector
    )
    logp_pi = distribution.log_prob(pi_action)
"""
tf_sample_code = """
sample_function()
"""
print("Tensorflow test...")
tf_time = timeit.timeit(tf_sample_code, setup=tf_setup_code, number=N_SAMPLE)

######################################################
# Print results ######################################
######################################################
print("\nTest Pytorch/Tensorflow rsample + log_prob + squash method speed:")
print(f"- Pytorch log_prob squash time: {pytorch_time} s")
print(f"- Tf log_prob squash time: {tf_time} s")
