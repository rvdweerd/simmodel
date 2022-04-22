import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import MessagePassing, GATv2Conv
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean, scatter_max
import copy

class EMB_LSTM(nn.Module):
    def __init__(self, state_dim, hp=None):
        super().__init__()
        self.hp = hp
        self.emb_dim = hp.emb_dim
        self.hidden_size = hp.hidden_size
        self.num_recurrent_layers = hp.recurrent_layers
        assert self.emb_dim == self.hidden_size
        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, num_layers = self.num_recurrent_layers)
        self.hidden_cell = None
             
    def reset_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size).to(device),
                            torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size).to(device))
        
    def forward(self, features, terminal=None):
        # features: (seq_len, num nodes in batch, feat_dim=emb_dim+3)
        #return features
        assert len(features.shape)==3
        seq_len = features.shape[0]
        max_nodes = self.hp.max_possible_nodes
        batch_size = features.shape[1]
        device = features.device

        if self.hidden_cell is None: #or max_nodes * batch_size != self.hidden_cell[0].shape[1]:
            self.reset_init_state(batch_size, device)
        if terminal is not None:           
            assert self.hidden_cell[0][:,terminal.bool(),:].sum() == 0
            assert self.hidden_cell[1][:,terminal.bool(),:].sum() == 0
   
        full_output, self.hidden_cell = self.lstm(features, self.hidden_cell)
        
        return full_output

class MaskablePPOPolicy_EMB_LSTM_SIMP(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev=None, hp=None):
        super().__init__()
        #assert hp.lstm_on
        self.description = 'Action Masked PPO Policy with GATv2 feature extraction and LSTM applied on node embeddings'
        
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space 
        hp.lstm_on = False # swith LSTM off in Actor/Critic modules
        self.hp=hp

        self.FE = FeatureExtractor(state_dim, hp)
        self.LSTM = EMB_LSTM(state_dim, hp)
        self.PI = Actor(state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev, hp=hp)
        self.V  = Critic(state_dim, hp, lstm=None)

    def forward(self, features, reachable, terminal=None):
        prob_dist = self.PI(features, reachable, terminal)
        values = self.V(features, terminal)
        return prob_dist, values

    def get_values(self, features, terminal=None):
        return self.V(features, terminal)

class FeatureExtractor(nn.Module):
    def __init__(self, state_dim, hp):
        super().__init__()
        self.hp = hp
        self.emb_dim = hp.emb_dim
        self.fc1 = nn.Linear(state_dim, self.emb_dim)
        
    def forward(self, state):
        # state: (seq_len, bsize, flatvecdim)
        assert len(state.shape)==3
        seq_len = state.shape[0]
        batch_size=state.shape[1]
        flatvecdim=state.shape[-1]
        device=state.device

        features = self.fc1(state)
        return features, 8*batch_size, None, 8

class FeatureExtractor_LSTM(nn.Module):
    def __init__(self, state_dim, hp):
        super().__init__()
        self.hp = hp
        self.emb_dim = hp.emb_dim
        self.hidden_size = hp.hidden_size
        self.num_recurrent_layers = hp.recurrent_layers
        self.T = 5  # Number of layers (/hops that yield a receptive field of the graph for each node
        self.node_dim = hp.node_dim
        self.lstm = nn.LSTM(self.node_dim, self.hidden_size, num_layers = self.num_recurrent_layers)
        kwargs={'concat':True}
        self.gat = GATv2(
            in_channels = self.hidden_size,
            hidden_channels = self.emb_dim,
            heads = 2,
            num_layers = self.T,
            out_channels = self.emb_dim,
            share_weights = True,
            **kwargs
        )
        self.hidden_cell = None

    def reset_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size).to(device),
                            torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size).to(device))

    def _deserialize(self, obs):
        # Deconstruct from an observation as defined by the PPO flattened observation wrapper
        # Args: obs [ nfm (NxF) | edge_list (Ex2) | reachable (N,) | num_nodes (1,) | max_num_nodes (1,) | num_edges (1,) | max_num_edges (1,) | node_dim (1,)]
        num_nodes, max_nodes, num_edges, max_edges, node_dim = obs[-5:].to(torch.int64).tolist()
        nfm , edge_index , reachable, _ = torch.split(obs,(self.hidden_size * max_nodes, 2 * max_edges, max_nodes, 5), dim=0)
        nfm = nfm.reshape(max_nodes,-1)[:num_nodes]
        edge_index = edge_index.reshape(2,-1)[:,:num_edges].to(torch.int64)
        reachable = reachable.reshape(-1,1).to(torch.int64)[:num_nodes]
        return nfm, edge_index, reachable, num_nodes, max_nodes, num_edges, max_edges, node_dim
        
    def forward(self, state):
        # state: (seq_len, bsize, flatvecdim)
        assert len(state.shape)==3
        seq_len = state.shape[0]
        batch_size=state.shape[1]
        device=state.device

        nfm_to_lstm = state[:,:,:(self.node_dim*self.hp.max_possible_nodes)].reshape(seq_len,-1,self.node_dim)
        assert nfm_to_lstm.shape[1] == self.hp.max_possible_nodes * batch_size
        full_output, self.hidden_cell = self.lstm(nfm_to_lstm, self.hidden_cell) # hidden cell defaults to zero tensors if None
        full_output = full_output.reshape(seq_len,batch_size,-1)
        assert full_output.shape[2] == self.hp.max_possible_nodes * self.hidden_size
        state_mem = torch.cat((full_output, state[:,:,(self.node_dim*self.hp.max_possible_nodes):]), dim=2)

        flatvecdim=state_mem.shape[-1]
        state_mem=state_mem.reshape(-1,flatvecdim)
        pyg_list=[]
        num_nodes_list=[]
        valid_entries_idx_list=[]
        reachable_nodes_tensor=torch.tensor([]).to(device)
        
        for i in range(batch_size * seq_len):
            pygx, pygei, reachable_nodes, num_nodes, max_nodes, num_edges, max_edges, node_dim = self._deserialize(state_mem[i])
                
            pyg_list.append(Data(
                pygx,
                pygei
            ))
            num_nodes_list.append(num_nodes)
            valid_entries_idx_list.append([i*max_nodes, i*max_nodes + num_nodes])
            reachable_nodes_tensor = torch.cat((reachable_nodes_tensor, reachable_nodes.squeeze()))
        pyg_data = Batch.from_data_list(pyg_list)
        
        # Apply Graph Attention Net to obtain latent node representations mu
        mu_raw = self.gat(pyg_data.x, pyg_data.edge_index) # (num_nodes_in_batch, emb_dim)
        valid_entries_idx = torch.tensor(valid_entries_idx_list, dtype=torch.int64)
        nodes_in_batch = pyg_data.num_nodes
        num_nodes = torch.tensor(num_nodes_list + [0]*(nodes_in_batch-batch_size*seq_len), dtype=torch.float, device=device)
        
        #assert sum(num_nodes_list) == pyg_data.batch.shape[0] == nodes_in_batch
        splitval = nodes_in_batch // seq_len
        decoupled_batch = pyg_data.batch[:splitval].repeat(seq_len) 
        num_nodes = num_nodes[:splitval].repeat(seq_len)
        features = torch.cat((mu_raw, decoupled_batch[:,None], reachable_nodes_tensor[:,None], num_nodes[:,None]), dim=1)
        
        return features.reshape(seq_len, nodes_in_batch // seq_len, self.emb_dim + 3), nodes_in_batch, valid_entries_idx, num_nodes

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev=None, hp=None):
        super().__init__()
        self.hp = hp
        self.emb_dim = hp.emb_dim
        self.action_dim = action_dim
        self.fc_pi = nn.Linear(self.emb_dim, self.action_dim, True)
        
    def forward(self, features, reachable, terminal=None):
        # features: (seq_len, num nodes in batch, feat_dim=emb_dim+3)
        assert len(features.shape)==3
        seq_len = features.shape[0]
        max_nodes = self.hp.max_possible_nodes
        num_nodes= features.shape[1]
        device = features.device
       
        prob_logits = self.fc_pi(features)
        if reachable is not None:
            prob_logits[~reachable] = -torch.inf
        policy_dist = distributions.Categorical(logits=prob_logits.to('cpu'))
        return policy_dist

class Critic(nn.Module):
    def __init__(self, state_dim, hp, lstm=None):
        super().__init__()
        self.emb_dim=hp.emb_dim
        self.hp=hp
        self.fc_v = nn.Linear(self.emb_dim, 1, True)        

    def forward(self, features, terminal=None):
        # features: [seq_len, bsize, feat_dim=emb_dim+3]
        seq_len = features.shape[0]
        max_nodes = self.hp.max_possible_nodes
        assert len(features.shape)==3
        #num_nodes_in_batch = features.shape[1]
        device = features.device

        v = self.fc_v(features)           
        return v#.unsqueeze(-1) # returns (seq_len, batch_size, 1)
