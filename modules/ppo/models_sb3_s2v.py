import argparse
import gym
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
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
from torch_scatter import scatter_mean, scatter_add
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Struc2VecExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Dict, emb_dim, emb_iter_T, node_dim):#, num_nodes):
        max_num_nodes=observation_space.spaces['W'].shape[0]
        features_dim: int = max_num_nodes * (emb_dim+1) #MUST BE NUM_NODES*(EMB_DIM+1), reacheble nodes vec concatenated
        super(Struc2VecExtractor, self).__init__(observation_space, features_dim)
        
        # Incoming: (bsize, num_nodes, (F+num_nodes+1))      
        # Compute shape by doing one forward pass
        #incoming_tensor_example = torch.as_tensor(observation_space.sample()[None]).float()
        self.norm_agg = True
        self.fdim           = features_dim
        self.emb_dim        = emb_dim
        self.T              = emb_iter_T
        self.node_dim       = node_dim
        self.theta1a    = nn.Linear(self.node_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta1b    = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta2     = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta3     = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta4     = nn.Linear(1, self.emb_dim, True)#, dtype=torch.float32)

    def get_embeddings(self, xv, Ws):
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

    def forward(self, observations):# -> torch.Tensor:
        num_nodes_padded = observations['W'].shape[1]
        #node_dim = observations['nfm'].shape[2]
        #bsize=observations['W'].shape[0]
        # if bsize>1:
        #     k=0
        
        nfm=observations['nfm'] # (bsize,numnodes,emb_dim)
        W=observations['W']     # (bsize,numnodes,numnodes)
        #pygei = observations['pygei'] #(bsize,2,MAX_NUM_EDGES)
        #pygx = observations['pygx'] #(bsize,numnodes,emb_dim)
        #reachable_nodes = observations['reachable_nodes'] #(bsize,numnodes,1)
        num_nodes=observations['num_nodes'] #(bsize,1)
        #num_edges=observations['num_edges'] #(bsize,1)

        mu = self.get_embeddings(nfm, W) # (batch_size, nr_nodes, emb_dim)      
        
        select=[]
        batch=[]
        #Reduce=False
        for i in range(len(num_nodes)):
            #lst+=[i]*n[i] + [len(n)]*(num_nodes_padded-n[i])
            select+=[True]*int(num_nodes[i])+[False]*(num_nodes_padded-int(num_nodes[i]))
            batch+=[i]*int(num_nodes[i])
        mu_raw = mu.reshape(-1,self.emb_dim)[select]
        batch=torch.tensor(batch,dtype=torch.int64,device=device)

        if self.norm_agg:
            mu_meanpool = scatter_mean(mu_raw,batch,dim=0) # (bsize,emb_dim)
            #mu_meanpool_expanded = torch.repeat_interleave(mu_meanpool,torch.tensor(num_nodes,dtype=torch.int64,device=device),dim=0) #(sum of num_nodes in the batch, emb_dim)
            #global_state_STAR = self.theta6(mu_meanpool_expanded)
        else:
            assert False
            mu_maxpool = scatter_add(mu_raw,batch,dim=0) # (bsize,emb_dim)
            #mu_maxpool_expanded = torch.repeat_interleave(mu_maxpool,torch.tensor(num_nodes,dtype=torch.int64,device=device),dim=0) #(sum of num_nodes in the batch, emb_dim)
            #global_state_STAR = self.theta6(mu_maxpool_expanded)


        mu_serialized1=torch.cat((mu,observations['reachable_nodes']),dim=2) #(bsize,numnodes,emb_dim+1)
        mu_serialized2=torch.cat((mu_meanpool,num_nodes),dim=1)[:,None,:] #(bsize,1,emb_dim+1)
        mu_serialized=torch.cat((mu_serialized1,mu_serialized2),dim=1) #(bsize,numnodes+1,emb_dim+1)

        # # test serialization
        # s0,s1,s2 = mu_serialized.shape
        # a,b=torch.split(mu_serialized,[s2-1,1],dim=2)
        # mu_deserialized, mu_mp_deserialized = torch.split(a,[s1-1,1],dim=1)
        # mu_mp_deserialized = mu_mp_deserialized.squeeze()
        # reachable_nodes_deserialized, num_nodes_deserialized = torch.split(b,[num_nodes_padded,1],dim=1)
        # num_nodes_deserialized = num_nodes_deserialized.squeeze(2)
        # assert s0 == bsize
        # assert (s2-1)==self.emb_dim
        # assert (s1-1)==num_nodes_padded
        # assert torch.allclose(mu,mu_deserialized)
        # assert torch.allclose(mu_meanpool,mu_mp_deserialized)
        # assert torch.allclose(reachable_nodes,reachable_nodes_deserialized)
        # assert torch.allclose(num_nodes,num_nodes_deserialized)

        # mu=torch.cat((mu,observations['reachable_nodes']),dim=2)
        # mu_flat = mu.reshape(bsize,-1)
        # assert self.fdim == self.features_dim
        # assert mu_flat.shape[-1] == self.fdim
        
        #return mu_flat
        return mu_serialized


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

    def _deserialize(self, features):
        s0,s1,s2 = features.shape
        a,b=torch.split(features,[s2-1,1],dim=2)
        mu, mu_mp = torch.split(a,[s1-1,1],dim=1)
        mu_mp = mu_mp.squeeze()
        reachable_nodes, num_nodes = torch.split(b,[s1-1,1],dim=1)
        num_nodes = num_nodes.squeeze(2)
        num_nodes_padded = s1-1
        emb_dim = s2-1
        bsize=s0
        return mu, mu_mp, reachable_nodes, num_nodes, num_nodes_padded, emb_dim, bsize

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        #num_nodes=features.shape[1]//(self.emb_dim+1)
        mu, mu_mp, reachable_nodes, num_nodes, num_nodes_padded, emb_dim, bsize = self._deserialize(features)
        if bsize>1:
            k=0
        return  self.forward_actor(  features, mu, num_nodes_padded), \
                self.forward_critic( features, mu, reachable_nodes, num_nodes_padded)

    def forward_actor(self, features: torch.Tensor, mu=None, num_nodes_padded=None) -> torch.Tensor:
        if mu==None:
            mu, mu_mp, reachable_nodes, num_nodes, num_nodes_padded, emb_dim, bsize = self._deserialize(features)

        global_state = self.theta6_pi(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes_padded, 1))
        local_action = self.theta7_pi(mu)  # (batch_dim, nr_nodes, emb_dim)
        rep = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (batch_dim, nr_nodes, 2*emb_dim)
        
        # Extra linear layer in sb3?
        prob_logits = self.theta5_pi(rep)#.squeeze(dim=2) # (batch_dim, nr_nodes)
        return prob_logits # (bsize, num_nodes,1)
        
        #return F.relu(prob_logits) # (bsize, num_nodes) TODO CHECK: extra nonlinearity useful?

    def forward_critic(self, features: torch.Tensor, mu=None, reachable_nodes=None, num_nodes_padded=None) -> torch.Tensor:
        if mu == None:
            mu, mu_mp, reachable_nodes, num_nodes, num_nodes_padded, emb_dim, bsize = self._deserialize(features)

        global_state = self.theta6_v(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes_padded, 1))
        local_action = self.theta7_v(mu)  # (batch_dim, nr_nodes, emb_dim)
        rep = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (batch_dim, nr_nodes, 2*emb_dim)
        qvals = self.theta5_v(rep).squeeze(-1) # (batch_dim, nr_nodes)
        reachable_nodes = reachable_nodes.type(torch.BoolTensor)
        qvals[~reachable_nodes.squeeze(-1)] = -torch.inf
        v=torch.max(qvals,dim=1)[0]
        return v.unsqueeze(-1) # (bsize,)


# from stable_baselines3.common.policies import BasePolicy
# from stable_baselines3.common.torch_layers import (
#     BaseFeaturesExtractor,
#     CombinedExtractor,
#     FlattenExtractor,
#     MlpExtractor,
#     NatureCNN,
#     )
# from stable_baselines3.common.type_aliases import Schedule
# from torch import nn
# import torch as th
# from functools import partial
# from sb3_contrib.common.maskable.distributions import MaskableDistribution, make_masked_proba_distribution
# from typing import Any, Dict, List, Optional, Tuple, Type, Union

# class MaskableActorCriticPolicy_nodeinvar(MaskableActorCriticPolicy):
#     def __init__(
#         self,
#         observation_space: gym.spaces.Space,
#         action_space: gym.spaces.Space,
#         lr_schedule: Schedule,
#         net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
#         activation_fn: Type[nn.Module] = nn.Tanh,
#         ortho_init: bool = True,
#         features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#     ):
#         super().__init__(
#             observation_space, 
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             ortho_init,
#             features_extractor_class,
#             features_extractor_kwargs,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs
#         )
#         # Default network architecture, from stable-baselines
#         if net_arch is None:
#             if features_extractor_class == NatureCNN:
#                 net_arch = []
#             else:
#                 #net_arch = [dict(pi=[64, 64], vf=[64, 64])]
#                 net_arch = []#dict(pi=[64, 64], vf=[64, 64])]

#         self.net_arch = net_arch
#         self.activation_fn = activation_fn
#         self.ortho_init = ortho_init

#         self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
#         self.features_dim = self.features_extractor.features_dim

#         self.normalize_images = normalize_images
#         # Action distribution
#         self.action_dist = make_masked_proba_distribution(action_space)

#         self._build(lr_schedule)

    
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
    def numTrainableParameters(self):
        print('Qnet size:')
        print('------------------------------------------')
        total = 0
        for name, p in self.named_parameters():
            total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("Total number of parameters: {}".format(total))
        print('------------------------------------------')
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

#from modules.ppo.models_sb3 import s2v_ACNetwork
class DeployablePPOPolicy(nn.Module):
    # implemented invariant to number of nodes
    def __init__(self, env, trained_policy):
        super(DeployablePPOPolicy, self).__init__()
        self.device=device
        self.struc2vec_extractor = Struc2VecExtractor(env.observation_space,64,5,env.F).to(device)
        self.struc2vec_extractor.load_state_dict(trained_policy.features_extractor.state_dict())
        
        self.s2vACnet = s2v_ACNetwork(64,1,1,64).to(device)
        self.s2vACnet.load_state_dict(trained_policy.mlp_extractor.state_dict())

        self.pnet = nn.Linear(1,1,True).to(device)
        self.pnet.load_state_dict(trained_policy.action_net.state_dict())

        self.vnet = nn.Linear(1,1,True).to(device)
        self.vnet.load_state_dict(trained_policy.value_net.state_dict())
        #Q_target.load_state_dict(policy.model.state_dict())

    def forward(self, obs):
        #obs = obs[None,:].to(device)
        y=self.struc2vec_extractor(obs)
        a,b=self.s2vACnet(y)
        logits=self.pnet(a)
        value=self.vnet(b)
        return logits, value

    def predict(self, obs, deterministic=True, action_masks=None):
        # obs comes in as (bsize,nodes,(V+F+1)), action masks as (nodes,)
        assert self.device == device
        #obs=obs.to(device)
        raw_logits, value = self.forward(obs)
        m=torch.as_tensor(action_masks, dtype=torch.bool, device=device).squeeze()
        HUGE_NEG = torch.tensor(-1e20, dtype=torch.float32, device=device)
        logits = torch.where(m,raw_logits.squeeze(),HUGE_NEG)
        if deterministic:
            action = torch.argmax(logits)
        else:
            assert False
        action = action.detach().cpu().numpy()
        return action, None

    def get_distribution(self, obs):
        # obs comes in as
        #return torch.categorical
        #obs=obs.to(device)
        raw_logits, value = self.forward(obs)


        m=obs['reachable_nodes'].to(torch.bool)
        #m=obs[:,:,-1].to(torch.bool)

        
        HUGE_NEG = torch.tensor(-1e20, dtype=torch.float32, device=device)
        prob_logits = torch.where(m.squeeze(), raw_logits.squeeze(-1), HUGE_NEG)
        distro=Categorical(logits=prob_logits)
        return distro

    def predict_values(self, obs):
        #obs=obs.to(device)
        raw_logits, value = self.forward(obs)
        return value

    def numTrainableParameters(self):
        print('Qnet size:')
        print('------------------------------------------')
        total = 0
        for name, p in self.named_parameters():
            total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("Total number of parameters: {}".format(total))
        print('------------------------------------------')
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total
