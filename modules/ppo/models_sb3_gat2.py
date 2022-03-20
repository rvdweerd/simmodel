import gym
import copy
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch_geometric.nn.conv import MessagePassing, GATv2Conv
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean, scatter_add
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GATv2(BasicGNN):
    r"""
    """
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:

        kwargs = copy.copy(kwargs)
        if 'heads' in kwargs and out_channels % kwargs['heads'] != 0:
            kwargs['heads'] = 1
        if 'concat' not in kwargs or kwargs['concat']:
            out_channels = out_channels // kwargs.get('heads', 1)

        return GATv2Conv(in_channels, out_channels, dropout=self.dropout,
                       **kwargs)

class Gat2Extractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Dict, emb_dim, emb_iter_T, node_dim):#, num_nodes):
        max_num_nodes=observation_space.spaces['W'].shape[0]
        features_dim: int = max_num_nodes * (emb_dim+1) #MUST BE NUM_NODES*(EMB_DIM+1), reacheble nodes vec concatenated
        super(Gat2Extractor, self).__init__(observation_space, features_dim)
        
        
        # Incoming: (bsize, num_nodes, (F+num_nodes+1))      
        # Compute shape by doing one forward pass
        #incoming_tensor_example = torch.as_tensor(observation_space.sample()[None]).float()
        self.norm_agg = True
        self.fdim           = features_dim
        self.emb_dim        = emb_dim
        self.T              = emb_iter_T
        self.node_dim       = node_dim
        kwargs={'concat':True}
        self.gat = GATv2(
            in_channels = self.node_dim,
            hidden_channels = self.emb_dim,
            num_layers = emb_iter_T,
            out_channels = self.emb_dim,
            **kwargs
        ).to(device)        

    def forward(self, observations):# -> torch.Tensor:
        num_nodes_padded = observations['W'].shape[1]
        #nfm=observations['nfm'] # (bsize,numnodes,emb_dim)
        #W=observations['W']     # (bsize,numnodes,numnodes)
        pygei = observations['pygei'] #(bsize,2,MAX_NUM_EDGES)
        pygx = observations['pygx'] #(bsize,numnodes,emb_dim)
        reachable_nodes = observations['reachable_nodes'] #(bsize,numnodes,1)
        num_nodes=observations['num_nodes'] #(bsize,1)
        num_edges=observations['num_edges'] #(bsize,1)
        assert num_edges.max() < pygei.shape[-1]
        bsize=len(num_nodes)
        if bsize>1:
            k=0
        # Build pyg data batch
        #X=[]
        #EI=[]
        select=[]
        pyg_list=[]
        for i in range(bsize):
            select+=[True]*int(num_nodes[i])+[False]*(num_nodes_padded-int(num_nodes[i]))
            pyg_list.append(Data(
                pygx[i][:int(num_nodes[i].detach().item()) ],
                pygei[i][:,:int(num_edges[i].detach().item())].to(torch.int64)
            ))
        pyg_data = Batch.from_data_list(pyg_list)
        mu_raw = self.gat(pyg_data.x, pyg_data.edge_index) # (nr_nodes in batch, emb_dim)      
        
        reach_nodes_enc = reachable_nodes.reshape(-1)[select][:,None]

        num_nodes_enc = torch.zeros(mu_raw.shape[0],1).to(device)
        num_nodes_enc[-1,0]=bsize
        num_nodes_enc[:bsize,0]=num_nodes.clone().squeeze()

        if self.norm_agg:
            mu_meanpool = scatter_mean(mu_raw,pyg_data.batch,dim=0) # (bsize,emb_dim)
            mu_meanpool_expanded = torch.repeat_interleave(mu_meanpool,num_nodes.squeeze().to(torch.int64),dim=0) #(sum of num_nodes in the batch, emb_dim)
            #global_state_STAR = self.theta6(mu_meanpool_expanded)
        else:
            assert False
        
        mu_serialized=torch.cat((mu_raw,mu_meanpool_expanded,num_nodes_enc,reach_nodes_enc),dim=1) #(num_nodes in batch, 2*emb_dim+2)

        return mu_serialized


class Gat2_ACNetwork(nn.Module):
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
        max_num_nodes: int=0,
    ):
        super(Gat2_ACNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.feature_dim = feature_dim
        self.emb_dim = emb_dim
        self.max_num_nodes = max_num_nodes

        # Policy network parameters
        self.theta5_pi = nn.Linear(2*self.emb_dim, 1, True)#, dtype=torch.float32)
        self.theta6_pi = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta7_pi = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)

        # Value network parameters        
        self.theta5_v = nn.Linear(2*self.emb_dim, 1, True)#, dtype=torch.float32)
        self.theta6_v = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta7_v = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)

    def _deserialize(self, features):
        mu_raw, mu_meanpool_expanded, num_nodes_enc, reach_nodes_enc = torch.split(features,[self.emb_dim,self.emb_dim,1,1],dim=1)
        bsize=int(num_nodes_enc[-1,0].detach().item())
        num_nodes=num_nodes_enc[:bsize,0]
        
        return mu_raw, mu_meanpool_expanded, num_nodes, reach_nodes_enc, bsize

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        #num_nodes=features.shape[1]//(self.emb_dim+1)
        mu_raw, mu_meanpool_expanded, num_nodes, reach_nodes, bsize = self._deserialize(features)
        if bsize>1:
            k=0
        return  self.forward_actor(  features, mu_raw, mu_meanpool_expanded, num_nodes, reach_nodes, bsize), \
                self.forward_critic( features, mu_raw, mu_meanpool_expanded, num_nodes, reach_nodes, bsize)

    def forward_actor(self, features: torch.Tensor, mu_raw=None, mu_meanpool_expanded=None, num_nodes=None, reach_nodes=None, bsize=None) -> torch.Tensor:
        if mu_raw==None:
            mu_raw, mu_meanpool_expanded, num_nodes, reach_nodes, bsize = self._deserialize(features)

        global_state = self.theta6_pi(mu_meanpool_expanded) # yields (#nodes in batch, emb_dim)
        local_action = self.theta7_pi(mu_raw)  # yields (#nodes in batch, emb_dim)
        rep = F.relu(torch.cat([global_state, local_action], dim=1)) # concat creates (#nodes in batch, 2*emb_dim)
        
        if bsize>1:
            k=0

        # Extra linear layer in sb3?
        prob_logits = self.theta5_pi(rep)#.squeeze(dim=2) # (batch_dim, nr_nodes)
        #max_num_nodes = int(num_nodes.max().detach().item())

        # add back batch dimension 
        splitter = num_nodes.detach().cpu().to(torch.int64).tolist()
        prob_logits = torch.nn.utils.rnn.pad_sequence(prob_logits.split(splitter, dim=0), batch_first=True, padding_value=-1e20) # (bsize, max_num_nodes, 1)
        p = self.max_num_nodes - prob_logits.shape[1]
        prob_logits = torch.nn.functional.pad(prob_logits,(0,0,0,p),value=-1e20)
        return prob_logits

    def forward_critic(self, features: torch.Tensor, mu_raw=None, mu_meanpool_expanded=None, num_nodes=None, reach_nodes=None, bsize=None) -> torch.Tensor:
        if mu_raw == None:
            mu_raw, mu_meanpool_expanded, num_nodes, reach_nodes, bsize = self._deserialize(features)

        global_state = self.theta6_pi(mu_meanpool_expanded) # yields (#nodes in batch, emb_dim)
        local_action = self.theta7_pi(mu_raw)  # yields (#nodes in batch, emb_dim)
        rep = F.relu(torch.cat([global_state, local_action], dim=1)) # concat creates (#nodes in batch, 2*emb_dim)     
        
        qvals = self.theta5_v(rep).squeeze(-1) # (batch_dim, nr_nodes)
        reachable_nodes = reach_nodes.squeeze().type(torch.BoolTensor)
        qvals[~reachable_nodes] = -1e20#torch.inf

        if bsize>1:
            k=0

        splitter = num_nodes.detach().cpu().to(torch.int64).tolist()
        qvals=torch.nn.utils.rnn.pad_sequence(qvals.split(splitter, dim=0), batch_first=True, padding_value=-1e20)
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

    
class Gat2_ActorCriticPolicy(MaskableActorCriticPolicy):
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

        super(Gat2_ActorCriticPolicy, self).__init__(
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
        self.mlp_extractor = Gat2_ACNetwork(self.features_dim, 1, 1, emb_dim, self.action_space.n)
        
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
class DeployablePPOPolicy_gat2(nn.Module):
    # implemented invariant to number of nodes
    def __init__(self, env, trained_policy):
        super(DeployablePPOPolicy_gat2, self).__init__()
        self.device=device
        self.gat2_extractor = Gat2Extractor(env.observation_space,64,5,env.F).to(device)
        self.gat2_extractor.load_state_dict(trained_policy.features_extractor.state_dict())
        
        self.gat2ACnet = Gat2_ACNetwork(64,1,1,64).to(device)
        self.gat2ACnet.load_state_dict(trained_policy.mlp_extractor.state_dict())

        self.pnet = nn.Linear(1,1,True).to(device)
        self.pnet.load_state_dict(trained_policy.action_net.state_dict())

        self.vnet = nn.Linear(1,1,True).to(device)
        self.vnet.load_state_dict(trained_policy.value_net.state_dict())
        #Q_target.load_state_dict(policy.model.state_dict())

    def forward(self, obs):
        #obs = obs[None,:].to(device)
        y=self.gat2_extractor(obs)
        a,b=self.gat2ACnet(y)
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
