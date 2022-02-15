import argparse
import gym
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import simdata_utils as su
from stable_baselines3.common.env_checker import check_env
#from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
#from stable_baselines3.common.policies import ActorCriticPolicy
#from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from modules.sim.simdata_utils import SimulateInteractiveMode
from modules.rl.environments import GraphWorld
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.rl_utils import EvaluatePolicy, print_parameters, GetFullCoverageSample, NpWrapper
#from stable_baselines3.common.vec_env import DummyVecEnv
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us, NFM_ev_ec_t_u
from modules.sim.graph_factory import GetWorldSet, LoadData
from modules.rl.environments import SuperEnv
from modules.ppo.ppo_wrappers import PPO_ActWrapper, PPO_ObsWrapper
#import modules.sim.simdata_utils as su
from modules.rl.rl_custom_worlds import GetCustomWorld
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Struc2Vec(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, emb_dim, emb_iter_T, node_dim):#, num_nodes):
        num_nodes=observation_space.shape[0]
        features_dim: int = num_nodes * (emb_dim+1) #MUST BE NUM_NODES*(EMB_DIM+1), reacheble nodes vec concatenated
        super(Struc2Vec, self).__init__(observation_space, features_dim)
        
        
        # Incoming: (bsize, num_nodes, (F+num_nodes+1))      
        # Compute shape by doing one forward pass
        incoming_tensor_example = torch.as_tensor(observation_space.sample()[None]).float()
        self.fdim = features_dim
        self.emb_dim        = emb_dim
        self.T              = emb_iter_T
        self.node_dim       = node_dim
        self.theta1a    = nn.Linear(self.node_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta1b    = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta2     = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta3     = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta4     = nn.Linear(1, self.emb_dim, True)#, dtype=torch.float32)

    def struc2vec(self, xv, Ws):
        # xv: The node features (batch_size, num_nodes, node_dim)
        # Ws: The graphs (batch_size, num_nodes, num_nodes)
        num_nodes = xv.shape[1]
        batch_size = xv.shape[0]
        
        # pre-compute 1-0 connection matrices masks (batch_size, num_nodes, num_nodes)
        #conn_matrices = torch.where(Ws > 0, torch.ones_like(Ws), torch.zeros_like(Ws)).to(device)
        conn_matrices = Ws # we have only edge weights of 1

        # Graph embedding
        # Note: we first compute s1 and s3 once, as they are not dependent on mu
        mu = torch.zeros(batch_size, num_nodes, self.emb_dim,device=device)#, dtype=torch.float32)#,device=device)
        #s1 = self.theta1a(xv)  # (batch_size, num_nodes, emb_dim)
        s1 = self.theta1b(F.relu(self.theta1a(xv)))  # (batch_size, num_nodes, emb_dim)
        #for layer in self.theta1_extras:
        #    s1 = layer(F.relu(s1))  # we apply the extra layer
        
        s3_1 = F.relu(self.theta4(Ws.unsqueeze(3)))  # (batch_size, nr_nodes, nr_nodes, emb_dim) - each "weigth" is a p-dim vector        
        s3_2 = torch.sum(s3_1, dim=1)  # (batch_size, nr_nodes, emb_dim) - the embedding for each node
        s3 = self.theta3(s3_2)  # (batch_size, nr_nodes, emb_dim)
        
        for t in range(self.T):
            s2 = self.theta2(conn_matrices.matmul(mu))    
            mu = F.relu(s1 + s2 + s3) # (batch_size, nr_nodes, emb_dim)
        
        return mu

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        num_nodes = observations.shape[1]
        node_dim = observations.shape[2]-1-num_nodes
        nfm, W, reachable_nodes = torch.split(observations,[node_dim, num_nodes, 1],2)
        mu = self.struc2vec(nfm, W) # (batch_size, nr_nodes, emb_dim)      
        mu=torch.cat((mu,reachable_nodes),dim=2)

        bsize=observations.shape[0]
        mu_flat = mu.reshape(bsize,-1)
        assert self.fdim == self.features_dim
        assert mu_flat.shape[-1] == self.fdim
        return mu_flat


class s2v_ACNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        emb_dim: int =0,
        #num_nodes: int=0,
    ):
        super(s2v_ACNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.feature_dim = feature_dim
        self.emb_dim = emb_dim
        #self.num_nodes = num_nodes

        # Policy network parameters
        self.theta5_pi = nn.Linear(2*self.emb_dim, 1, True)#, dtype=torch.float32)
        self.theta6_pi = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta7_pi = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)

        # Value network parameters        
        self.theta5_v = nn.Linear(2*self.emb_dim, 1, True)#, dtype=torch.float32)
        self.theta6_v = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta7_v = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)


    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        num_nodes=features.shape[1]//(self.emb_dim+1)
        return self.forward_actor(features,num_nodes), self.forward_critic(features,num_nodes)

    def forward_actor(self, features: torch.Tensor, num_nodes: int) -> torch.Tensor:
        #num_nodes = features.shape[1]//(self.emb_dim+1)
        mu_rn = features.reshape(features.shape[0], num_nodes, self.emb_dim+1)  # (batch_size, nr_nodes, emb_dim+1)
        mu, reachable_nodes = torch.split(mu_rn,[self.emb_dim, 1],2)

        global_state = self.theta6_pi(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1))
        local_action = self.theta7_pi(mu)  # (batch_dim, nr_nodes, emb_dim)
        rep = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (batch_dim, nr_nodes, 2*emb_dim)
        
        # Extra linear layer in sb3?
        prob_logits = self.theta5_pi(rep)#.squeeze(dim=2) # (batch_dim, nr_nodes)
        return prob_logits # (bsize, num_nodes,1)
        
        #return F.relu(prob_logits) # (bsize, num_nodes) TODO CHECK: extra nonlinearity useful?

    def forward_critic(self, features: torch.Tensor, num_nodes: int) -> torch.Tensor:
        #mu = features.reshape(-1, self.num_nodes, self.emb_dim)  # (batch_size, nr_nodes, emb_dim)
        mu_rn = features.reshape(-1, num_nodes, self.emb_dim+1)  # (batch_size, nr_nodes, emb_dim+1)
        mu, reachable_nodes = torch.split(mu_rn,[self.emb_dim, 1],2)

        global_state = self.theta6_v(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1))
        local_action = self.theta7_v(mu)  # (batch_dim, nr_nodes, emb_dim)
        rep = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (batch_dim, nr_nodes, 2*emb_dim)
        qvals = self.theta5_v(rep).squeeze(-1) # (batch_dim, nr_nodes)
        reachable_nodes = reachable_nodes.type(torch.BoolTensor)
        qvals[~reachable_nodes.squeeze(-1)] = -torch.inf
        v=torch.max(qvals,dim=1)[0]
        return v.unsqueeze(-1) # (bsize,)


from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    )
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
import torch as th
from functools import partial
from sb3_contrib.common.maskable.distributions import MaskableDistribution, make_masked_proba_distribution
from typing import Any, Dict, List, Optional, Tuple, Type, Union

class MaskableActorCriticPolicy_nodeinvar(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space, 
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs
        )
        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                #net_arch = [dict(pi=[64, 64], vf=[64, 64])]
                net_arch = []#dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        # Action distribution
        self.action_dist = make_masked_proba_distribution(action_space)

        self._build(lr_schedule)

    



class s2v_ActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        #max_num_nodes=5,
        *args,
        **kwargs,
    ):

        super(s2v_ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.info = net_arch
        #self.max_num_nodes=max_num_nodes

    def _build_mlp_extractor(self) -> None:
        emb_dim=self.features_extractor_kwargs['emb_dim']
        node_dim=self.features_extractor_kwargs['node_dim']
        #max_num_nodes=self.features_extractor_kwargs['num_nodes']
        self.mlp_extractor = s2v_ACNetwork(self.features_dim, 1, 1, emb_dim)# max_num_nodes)
        
        # emb_dim = self.features_extractor_kwargs['emb_dim']
        # emb_iter_T = self.features_extractor_kwargs['emb_iter_T']
        # node_dim =  self.features_extractor_kwargs['node_dim']
        # num_nodes = emb_dim = self.features_extractor_kwargs['num_nodes']
        # self.mlp_extractor = s2v_ACNetwork(self.features_dim, num_nodes, 1, emb_dim, num_nodes)        
        #self.net_arch['max_num_nodes']