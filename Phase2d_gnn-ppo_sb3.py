import argparse
import gym
from modules.sim.simdata_utils import SimulateInteractiveMode
import simdata_utils as su
from stable_baselines3.common.env_checker import check_env
from modules.rl.environments import GraphWorld
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.rl_utils import EvaluatePolicy, print_parameters, GetFullCoverageSample, NpWrapper
#from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us, NFM_ev_ec_t_u
from modules.ppo.ppo_wrappers import PPO_ActWrapper, PPO_ObsWrapper
#import modules.sim.simdata_utils as su
from modules.rl.rl_custom_worlds import GetCustomWorld
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

device = 'cuda' if torch.cuda.is_available() else 'cpu'

nfm_funcs = {
    'NFM_ev_ec_t'       : NFM_ev_ec_t(),
    'NFM_ec_t'          : NFM_ec_t(),
    'NFM_ev_t'          : NFM_ev_t(),
    'NFM_ev_ec_t_u'     : NFM_ev_ec_t_u(),
    'NFM_ev_ec_t_um_us' : NFM_ev_ec_t_um_us(),
    }


def CreateEnv(max_nodes=9):
    world_name='Manhattan3x3_WalkAround'
    state_repr='etUte0U0'
    state_enc='nfm'
    nfm_func_name = 'NFM_ev_ec_t'
    edge_blocking = True
    remove_world_pool = True
    env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    env.redefine_nfm(nfm_funcs[nfm_func_name])
    env.capture_on_edges = edge_blocking
    if remove_world_pool:
        env._remove_world_pool()
    #SimulateInteractiveMode(env,filesave_with_time_suffix=False)
    MAX_NODES=9
    env = PPO_ObsWrapper(env, max_possible_num_nodes = max_nodes)        
    env = PPO_ActWrapper(env)        

    return env

MAX_NODES=9
env=CreateEnv(max_nodes=MAX_NODES)
EMB_DIM = 16
EMB_ITER_T = 3
NODE_DIM = env.F

obs=env.reset()
k=0





from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
class Struc2Vec(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, emb_dim, emb_iter_T, node_dim, num_nodes):
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


from stable_baselines3.common.policies import ActorCriticPolicy
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
        num_nodes: int=0,
    ):
        super(s2v_ACNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.feature_dim = feature_dim
        self.emb_dim = emb_dim
        self.num_nodes = num_nodes

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
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        mu_rn = features.reshape(-1, self.num_nodes, self.emb_dim+1)  # (batch_size, nr_nodes, emb_dim+1)
        mu, reachable_nodes = torch.split(mu_rn,[self.emb_dim, 1],2)

        global_state = self.theta6_pi(torch.sum(mu, dim=1, keepdim=True).repeat(1, self.num_nodes, 1))
        local_action = self.theta7_pi(mu)  # (batch_dim, nr_nodes, emb_dim)
        rep = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (batch_dim, nr_nodes, 2*emb_dim)
        prob_logits = self.theta5_pi(rep).squeeze(dim=2) # (batch_dim, nr_nodes)
        # mask invalid actions
        #reachable_nodes = reachable_nodes.squeeze(dim=2).type(torch.BoolTensor)
        #prob_logits[~reachable_nodes] = -torch.inf
        return prob_logits # (bsize, num_nodes)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        #mu = features.reshape(-1, self.num_nodes, self.emb_dim)  # (batch_size, nr_nodes, emb_dim)
        mu_rn = features.reshape(-1, self.num_nodes, self.emb_dim+1)  # (batch_size, nr_nodes, emb_dim+1)
        mu, reachable_nodes = torch.split(mu_rn,[self.emb_dim, 1],2)

        global_state = self.theta6_v(torch.sum(mu, dim=1, keepdim=True).repeat(1, self.num_nodes, 1))
        local_action = self.theta7_v(mu)  # (batch_dim, nr_nodes, emb_dim)
        rep = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (batch_dim, nr_nodes, 2*emb_dim)
        qvals = self.theta5_v(rep).squeeze(-1) # (batch_dim, nr_nodes)
        reachable_nodes = reachable_nodes.type(torch.BoolTensor)
        qvals[~reachable_nodes.squeeze(-1)] = -torch.inf
        v=torch.max(qvals,dim=1)[0]
        return v # (bsize,)

class s2v_ActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
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

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = s2v_ACNetwork(self.features_dim, self.net_arch['max_num_nodes'], 1, self.net_arch['emb_dim'], self.net_arch['num_nodes'])
        #self.net_arch['max_num_nodes']
policy_kwargs = dict(
    features_extractor_class=Struc2Vec,
    features_extractor_kwargs=dict(emb_dim=EMB_DIM, emb_iter_T=EMB_ITER_T, node_dim=NODE_DIM, num_nodes=MAX_NODES),
    net_arch=dict(max_num_nodes=MAX_NODES, emb_dim=EMB_DIM, num_nodes=MAX_NODES)
    
)

model = PPO(s2v_ActorCriticPolicy, env, \
    #learning_rate=1e-4,\
    seed=0,\
    #clip_range=0.1,\    
    #max_grad_norm=0.1,\
    policy_kwargs = policy_kwargs, verbose=1, tensorboard_log="results/gnn-ppo/sb3/test/tensorboard/")

print_parameters(model.policy)
model.learn(total_timesteps = 10000)

