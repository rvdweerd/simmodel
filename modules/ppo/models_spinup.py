#import numpy as np
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
#from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class Struc2VecActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, emb_dim=8, emb_iter_T=3):
        super().__init__()
        self.emb_dim        = emb_dim
        self.T              = emb_iter_T
        obs_dim = observation_space.shape
        self.node_dim       = obs_dim[1]-obs_dim[0]-1
        # policy builder depends on action space
        # if isinstance(action_space, Box):
        #     self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        # elif isinstance(action_space, Discrete):
        #     self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # # build value function
        # self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]