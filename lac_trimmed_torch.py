"""A trimmed down version of my RL algorithm such that I can investigate why the full
training loop is slower in Pytorch compared to Tensorflow 2.x (eager mode).
"""

from copy import deepcopy

import torch
from torch.optim import Adam
import torch.nn.functional as F

from gaussian_actor_torch import SquashedGaussianActor
from lyapunov_critic_torch import LyapunovCritic

# Script parameters
GAMMA = 0.999
ALPHA = 0.99
ALPHA3 = 0.2
LABDA = 0.99
POLYAK = 5e-3
LR_A = 1e-4
LR_L = 3e-4
LR_LAG = 3e-4
OBS_DIM = 8
ACT_DIM = 3


class LAC(object):
    """The Lyapunov Actor Critic.
    """

    def __init__(self):

        # Create attributes
        self.LR_A = LR_A
        self.LR_L = LR_L
        self.LR_lag = LR_LAG
        self.polyak = POLYAK
        self.target_entropy = -ACT_DIM  # lower bound of the policy entropy

        # Create Gaussian Actor (GA) and Lyapunov critic (LC) Networks
        self.ga = SquashedGaussianActor(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            hidden_sizes=[64, 64],
            log_std_min=-20,
            log_std_max=2,
        )
        self.lc = LyapunovCritic(
            obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_sizes=[128, 128]
        )

        # Create GA and LC target networks
        # Don't get optimized but get updated according to the EMA of the main
        # networks
        self.ga_ = deepcopy(self.ga)
        self.lc_ = deepcopy(self.lc)

        # Freeze target networks
        for p in self.ga_.parameters():
            p.requires_grad = False
        for p in self.lc_.parameters():
            p.requires_grad = False

        # Create lagrance multiplier placeholders
        self.log_alpha = torch.tensor(ALPHA, dtype=torch.float32).log()
        self.log_alpha.requires_grad = True  # Enable gradient computation
        self.log_labda = torch.tensor(LABDA, dtype=torch.float32).log()
        self.log_labda.requires_grad = True  # Enable gradient computation

        ###########################################
        # Create optimizers #######################
        ###########################################
        self.alpha_train = Adam([self.log_alpha], lr=self.LR_A)
        self.a_train = Adam(self.ga.parameters(), lr=self.LR_A)
        self.lambda_train = Adam([self.log_labda], lr=self.LR_lag)
        self.l_train = Adam(self.lc.parameters(), lr=self.LR_L)

    def learn(self, batch):
        """Runs the SGD to update all the optimize parameters.

        Args:
            batch (numpy.ndarray): The batch of experiences.

        Returns:
            Tuple: Tuple with diagnostic information.
        """

        # Retrieve state, action and reward from the batch
        bs = batch["s"]  # state
        ba = batch["a"]  # action
        br = batch["r"]  # reward
        bterminal = batch["terminal"]
        bs_ = batch["s_"]  # next state

        # Update target networks
        self.update_target()

        # Calculate variables from which we do not require the gradients
        with torch.no_grad():
            a_, _, _ = self.ga_(bs_)
            l_ = self.lc_(bs_, a_)
            l_target = br + GAMMA * (1 - bterminal) * l_.detach()

        # Calculate current lyapunov value
        l = self.lc(bs, ba)

        # Calculate current value and target lyapunov multiplier value
        lya_a_, _, _ = self.ga(bs_)
        lya_l_ = self.lc(bs_, lya_a_)

        # Calculate log probability of a_input based on current policy
        pi, _, log_pis = self.ga(bs)

        # Calculate Lyapunov constraint function
        self.l_delta = torch.mean(lya_l_ - l.detach() + ALPHA3 * br)

        # Zero gradients on labda
        self.lambda_train.zero_grad()

        # Lagrance multiplier loss functions and optimizers graphs
        labda_loss = -torch.mean(self.log_labda * self.l_delta.detach())

        # Apply gradients to log_lambda
        labda_loss.backward()
        self.lambda_train.step()

        # Zero gradients on alpha
        self.alpha_train.zero_grad()

        # Calculate alpha loss
        alpha_loss = -torch.mean(
            self.log_alpha * log_pis.detach() + self.target_entropy
        )

        # Apply gradients
        alpha_loss.backward()
        self.alpha_train.step()

        # = Optimize actpr =
        # Zero gradients on the actor
        self.a_train.zero_grad()

        # Calculate actor loss
        a_loss = self.labda.detach() * self.l_delta + self.alpha.detach() * torch.mean(
            log_pis
        )

        # Apply gradients
        a_loss.backward()
        self.a_train.step()

        # = Optimize critic =
        self.l_train.zero_grad()

        # Calculate L_backup
        l_error = F.mse_loss(l_target, l)

        # Apply gradients
        l_error.backward()
        self.l_train.step()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def labda(self):
        return torch.clamp(self.log_labda.exp(), 0, 1)

    def update_target(self):
        # Polyak averaging for target variables
        with torch.no_grad():
            for pi_main, pi_targ in zip(self.ga.parameters(), self.ga_.parameters()):
                pi_targ.data.mul_(self.polyak)
                pi_targ.data.add_((1 - self.polyak) * pi_main.data)
            for pi_main, pi_targ in zip(self.lc.parameters(), self.lc_.parameters()):
                pi_targ.data.mul_(self.polyak)
                pi_targ.data.add_((1 - self.polyak) * pi_main.data)
