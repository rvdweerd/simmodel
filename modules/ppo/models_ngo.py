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

class GATv2(BasicGNN):
    """
    Basic Graph Attention Module (more expressive version 2)
    https://arxiv.org/abs/2105.14491
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

class EMB_LSTM(nn.Module):
    def __init__(self, state_dim, hp=None):
        super().__init__()
        #self.counter = 0
        #assert hp.lstm_on
        #self.lstm_on = hp.lstm_on
        self.hp = hp
        self.emb_dim = hp.emb_dim
        #self.action_dim = action_dim
        self.hidden_size = hp.hidden_size
        self.num_recurrent_layers = hp.recurrent_layers
        assert self.emb_dim == self.hidden_size
        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, num_layers = self.num_recurrent_layers)
        self.hidden_cell = None
             
    def reset_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size).to(device),
                            torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size).to(device))
        
    def forward(self, features_raw, terminal=None, selector=None):
        # features: (seq_len, num nodes in batch, feat_dim=emb_dim+3)
        assert len(features_raw.shape)==3
        seq_len = features_raw.shape[0]
        max_nodes = self.hp.max_possible_nodes
        num_nodes= features_raw.shape[1]
        device = features_raw.device

        mu_raw, batch, reachable, num_nodes_list_orig = torch.split(features_raw,[self.emb_dim,1,1,1],dim=2)
        batch_size = int(batch.max()) + 1 
        
        #batch = batch.squeeze(-1).to(torch.int64) 
        #reachable = reachable.squeeze(-1).bool() 

        if self.hidden_cell is None or max_nodes * batch_size != self.hidden_cell[0].shape[1]:
            self.reset_init_state(max_nodes * batch_size, device)
        if terminal is not None:
            # Set node cells of graph of the termined episode to 0
            terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(batch_size,dtype=torch.int64)*max_nodes, dim=0)
            assert self.hidden_cell[0][:,terminal_dense,:].sum() == 0
            assert self.hidden_cell[1][:,terminal_dense,:].sum() == 0

        assert selector is not None
        # num_nodes_list = num_nodes_list_orig[0].squeeze()
        # if selector == None:
        #     selector_=[]
        #     for i in range(batch_size):
        #         selector_+=[True]*int(num_nodes_list[i]) + [False]*(max_nodes-int(num_nodes_list[i]))
        #     selector=torch.tensor(selector_,dtype=torch.bool)   
        
        full_output, out_hidden_cell = self.lstm(mu_raw, (self.hidden_cell[0][:,selector,:], self.hidden_cell[1][:,selector,:]))
        self.hidden_cell[0][:,selector,:] = out_hidden_cell[0]
        self.hidden_cell[1][:,selector,:] = out_hidden_cell[1]
        mu_mem = full_output # (seq_len, num_nodes in batch, emb_dim)
        features_mem = torch.cat((mu_mem,batch, reachable, num_nodes_list_orig),dim=2)
        return features_mem

class MaskablePPOPolicy_EMB_LSTM(nn.Module):
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
        #print(self)
        #self.numTrainableParameters()

    def forward(self, features, terminal=None, selector=None):
        prob_dist = self.PI(features, terminal, selector)
        values = self.V(features, terminal, selector)
        return prob_dist, values

    def get_values(self, features, terminal=None, selector=None):
        return self.V(features, terminal, selector)

    def numTrainableParameters(self):
        ps=""
        ps+=self.description+'\n'
        ps+='------------------------------------------\n'
        total = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                total += np.prod(p.shape)
            ps+=("{:24s} {:12s} requires_grad={}\n".format(name, str(list(p.shape)), p.requires_grad))
        ps+=("Total number of trainable parameters: {}\n".format(total))
        ps+='------------------------------------------'
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, ps

class MaskablePPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev=None, hp=None):
        super().__init__()
        if hp.lstm_on:            
            self.description = 'Action Masked PPO Policy with A+C LSTMs and GATv2 feature extraction'
        else:
            self.description = 'Action Masked PPO Policy with LSTM switched off and GATv2 feature extraction'        
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space 
        self.hp=hp

        self.FE = FeatureExtractor(state_dim, hp)
        self.PI = Actor(state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev, hp=hp)
        self.V  = Critic(state_dim,hp)
        #print(self)
        #self.numTrainableParameters()

    def forward(self, features, terminal=None, selector=None):
        return self.PI(features, terminal, selector), self.V(features, terminal, selector)

    def get_values(self, features, terminal=None, selector=None):
        return self.V(features, terminal, selector)

    def numTrainableParameters(self):
        ps=""
        ps+=self.description+'\n'
        ps+='------------------------------------------\n'
        total = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                total += np.prod(p.shape)
            ps+=("{:24s} {:12s} requires_grad={}\n".format(name, str(list(p.shape)), p.requires_grad))
        ps+=("Total number of trainable parameters: {}\n".format(total))
        ps+='------------------------------------------'
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, ps

class MaskablePPOPolicy_CONCAT(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev=None, hp=None):
        super().__init__()
        assert hp.lstm_on
        self.description = 'Action Masked PPO Policy with A+C LSTMs on concatenated embeddings and GATv2 feature extraction'
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space 
        self.hp=hp

        self.FE = FeatureExtractor(state_dim, hp)
        self.PI = Actor_concat(state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev, hp=hp)
        self.V  = Critic_concat(state_dim,hp)
        #print(self)
        #self.numTrainableParameters()

    def forward(self, features, terminal=None, selector=None):
        return self.PI(features, terminal, selector), self.V(features, terminal, selector)

    def get_values(self, features, terminal=None, selector=None):
        return self.V(features, terminal, selector)

    def numTrainableParameters(self):
        ps=""
        ps+=self.description+'\n'
        ps+='------------------------------------------\n'
        total = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                total += np.prod(p.shape)
            ps+=("{:24s} {:12s} requires_grad={}\n".format(name, str(list(p.shape)), p.requires_grad))
        ps+=("Total number of trainable parameters: {}\n".format(total))
        ps+='------------------------------------------'
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, ps

class MaskablePPOPolicy_FE_LSTM(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev=None, hp=None):
        super().__init__()
        self.description = 'Action Masked PPO Policy with LSTM (on node features) + GATv2 feature extraction'
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space 
        hp.lstm_on = False # LSTM is not used for Actor and Critic
        self.hp=hp

        self.FE = FeatureExtractor_LSTM(state_dim, hp)
        self.PI = Actor(state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev, hp=hp)
        self.V  = Critic(state_dim,hp)
        #print(self)
        #self.numTrainableParameters()

    def forward(self, features, terminal=None, selector=None):
        return self.PI(features, terminal, selector), self.V(features, terminal, selector)

    def get_values(self, features, terminal=None, selector=None):
        return self.V(features, terminal, selector)

    def numTrainableParameters(self):
        ps=""
        ps+=self.description+'\n'
        ps+='------------------------------------------\n'
        total = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                total += np.prod(p.shape)
            ps+=("{:24s} {:12s} requires_grad={}\n".format(name, str(list(p.shape)), p.requires_grad))
        ps+=("Total number of trainable parameters: {}\n".format(total))
        ps+='------------------------------------------'
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, ps
  
class FeatureExtractor(nn.Module):
    def __init__(self, state_dim, hp):
        super().__init__()
        self.hp = hp
        self.emb_dim = hp.emb_dim
        self.T = 5  # Number of layers (/hops that yield a receptive field of the graph for each node
        self.node_dim = hp.node_dim
        kwargs={'concat':True}
        self.gat = GATv2(
            in_channels = self.node_dim,
            hidden_channels = self.emb_dim,
            heads = 1,#2,
            num_layers = self.T,
            out_channels = self.emb_dim,
            share_weights = False,#True,
            **kwargs
        )

    def _deserialize(self, obs):
        # Deconstruct from an observation as defined by the PPO flattened observation wrapper
        # Args: obs [ nfm (NxF) | edge_list (Ex2) | reachable (N,) | num_nodes (1,) | max_num_nodes (1,) | num_edges (1,) | max_num_edges (1,) | node_dim (1,)]
        num_nodes, max_nodes, num_edges, max_edges, node_dim = obs[-5:].to(torch.int64).tolist()
        nfm , edge_index , reachable, _ = torch.split(obs,(node_dim * max_nodes, 2 * max_edges, max_nodes, 5), dim=0)
        nfm = nfm.reshape(max_nodes,-1)[:num_nodes]
        edge_index = edge_index.reshape(2,-1)[:,:num_edges].to(torch.int64)
        reachable = reachable.reshape(-1,1).to(torch.int64)[:num_nodes]
        return nfm, edge_index, reachable, num_nodes, max_nodes, num_edges, max_edges, node_dim
        
    def forward(self, state):
        # state: (seq_len, bsize, flatvecdim)
        assert len(state.shape)==3
        seq_len = state.shape[0]
        batch_size=state.shape[1]
        flatvecdim=state.shape[-1]
        device=state.device

        state=state.reshape(-1,flatvecdim)
        pyg_list=[]
        num_nodes_list=[]
        #valid_entries_idx_list=[]
        reachable_nodes_tensor=torch.tensor([]).to(device)
        
        for i in range(batch_size * seq_len):
            pygx, pygei, reachable_nodes, num_nodes, max_nodes, num_edges, max_edges, node_dim = self._deserialize(state[i])
            pyg_list.append(Data(
                pygx,
                pygei
            ))
            num_nodes_list.append(num_nodes)
            #valid_entries_idx_list.append([i*max_nodes, i*max_nodes + num_nodes])
            reachable_nodes_tensor = torch.cat((reachable_nodes_tensor, reachable_nodes.squeeze()))
        pyg_data = Batch.from_data_list(pyg_list)
        
        # Apply Graph Attention Net to obtain latent node representations mu
        mu_raw = self.gat(pyg_data.x, pyg_data.edge_index) # (num_nodes_in_batch, emb_dim)
        #valid_entries_idx = torch.tensor(valid_entries_idx_list, dtype=torch.int64)
        nodes_in_batch = pyg_data.num_nodes
        num_nodes = torch.tensor(num_nodes_list + [0]*(nodes_in_batch-batch_size*seq_len), dtype=torch.float, device=device)
        
        #assert sum(num_nodes_list) == pyg_data.batch.shape[0] == nodes_in_batch
        splitval = nodes_in_batch // seq_len
        decoupled_batch = pyg_data.batch[:splitval].repeat(seq_len) 
        num_nodes = num_nodes[:splitval].repeat(seq_len)
        features = torch.cat((mu_raw, decoupled_batch[:,None], reachable_nodes_tensor[:,None], num_nodes[:,None]), dim=1)
        
        return features.reshape(seq_len, nodes_in_batch // seq_len, self.emb_dim + 3), nodes_in_batch, None, num_nodes

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
        #self.counter = 0
        self.lstm_on = hp.lstm_on
        self.hp = hp
        self.emb_dim = hp.emb_dim
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space 
        self.hidden_size = hp.hidden_size
        self.num_recurrent_layers = hp.recurrent_layers
        
        self.theta5_pi = nn.Linear(2*self.emb_dim, 1, True)
        if self.lstm_on:
            self.lstm = nn.LSTM(self.emb_dim, self.hidden_size, num_layers = self.num_recurrent_layers)
            self.theta6_pi = nn.Linear(  self.hidden_size, self.emb_dim, True)
            self.theta7_pi = nn.Linear(  self.hidden_size, self.emb_dim, True)
        else:
            self.theta6_pi = nn.Linear(  self.emb_dim, self.emb_dim, True)
            self.theta7_pi = nn.Linear(  self.emb_dim, self.emb_dim, True)            
        self.log_std_dev = nn.Parameter(init_log_std_dev * torch.ones(33, dtype=torch.float), requires_grad=trainable_std_dev)
        self.covariance_eye = torch.eye(33).unsqueeze(0)
        self.hidden_cell = None
        
        #print('Actor network:')
        #self.numTrainableParameters()
        #print(self)
        
    def reset_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size).to(device),
                            torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size).to(device))
        
    def forward(self, features, terminal=None, selector=None):
        # features: (seq_len, num nodes in batch, feat_dim=emb_dim+3)
        assert len(features.shape)==3
        seq_len = features.shape[0]
        max_nodes = self.hp.max_possible_nodes
        num_nodes= features.shape[1]
        device = features.device

        mu_raw, batch, reachable, num_nodes_list = torch.split(features,[self.emb_dim,1,1,1],dim=2)
        batch_size = int(batch.max()) + 1 
        
        batch = batch.squeeze(-1).to(torch.int64) 
        reachable = reachable.squeeze(-1).bool() 

        if self.hidden_cell is None or max_nodes * batch_size != self.hidden_cell[0].shape[1]:
            self.reset_init_state(max_nodes * batch_size, device)
        if terminal is not None:
            # Set node cells of graph of the termined episode to 0
            terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(batch_size,dtype=torch.int64)*max_nodes, dim=0)
            assert self.hidden_cell[0][:,terminal_dense,:].sum() == 0
            assert self.hidden_cell[1][:,terminal_dense,:].sum() == 0

        num_nodes_list = num_nodes_list[0].squeeze()
        if self.lstm_on:
            if selector == None:
                selector_=[]
                for i in range(batch_size):
                    selector_+=[True]*int(num_nodes_list[i]) + [False]*(max_nodes-int(num_nodes_list[i]))
                selector=torch.tensor(selector_,dtype=torch.bool)   

            full_output, out_hidden_cell = self.lstm(mu_raw, (self.hidden_cell[0][:,selector,:], self.hidden_cell[1][:,selector,:]))
            self.hidden_cell[0][:,selector,:] = out_hidden_cell[0]
            self.hidden_cell[1][:,selector,:] = out_hidden_cell[1]
            mu_mem = full_output # (seq_len, num_nodes in batch, emb_dim)
        else:
            mu_mem = mu_raw

        mu_mem_meanpool = scatter_mean(mu_mem, batch, dim=1) # (seq_len, batch_size, emb_dim)
        expander = num_nodes_list[:batch_size].to(torch.int64)   
        mu_mem_meanpool_expanded = torch.repeat_interleave(mu_mem_meanpool, expander, dim=1) # (num_nodes in batch, emb_dim)
        
        # Transform node embeddings to action log probabilities
        global_state = self.theta6_pi(mu_mem_meanpool_expanded) # yields (#nodes in batch, emb_dim)
        local_action = self.theta7_pi(mu_mem)  # yields (#nodes in batch, emb_dim)
        rep = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (#nodes in batch, 2*emb_dim)        
        prob_logits_raw = self.theta5_pi(rep).squeeze(-1) # (nr_nodes in batch,)

        # Apply action masking and pad to standard size (based on maximum Graph size in nodes in the trainset)
        prob_logits_raw[~reachable] = -torch.inf
        splitter = num_nodes_list[:batch_size].to(torch.int64).tolist()
        if prob_logits_raw.shape[0]==1:
            prob_logits_splitted = list(torch.split(prob_logits_raw.squeeze(), splitter, dim=0))
            p = max_nodes - prob_logits_splitted[0].shape[-1]
            prob_logits_splitted[0] = torch.nn.functional.pad(prob_logits_splitted[0], (0,p), value = -torch.inf)
            prob_logits_splitted = torch.nn.utils.rnn.pad_sequence(prob_logits_splitted, batch_first=True, padding_value = -torch.inf)
            prob_logits=prob_logits_splitted[None,:,:]
        else:
            prob_logits_splitted = list(torch.split(prob_logits_raw.squeeze(), splitter, dim=1))
            for i in range(len(prob_logits_splitted)):
                p = max_nodes - prob_logits_splitted[i].shape[-1]
                prob_logits_splitted[i] = torch.nn.functional.pad(prob_logits_splitted[i], (0,p), value=-torch.inf)
            prob_logits=torch.stack(prob_logits_splitted).permute((1,0,2))
       
        if self.continuous_action_space:
            assert False # not implemented
        else:
            policy_dist = distributions.Categorical(logits=prob_logits.to('cpu'))
        return policy_dist

    def numTrainableParameters(self):
        print('Actor size:')
        print('------------------------------------------')
        total = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("Total number of trainable parameters: {}".format(total))
        print('------------------------------------------')
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

class Critic(nn.Module):
    def __init__(self, state_dim, hp, lstm=None):
        super().__init__()
        self.lstm_on = hp.lstm_on
        self.emb_dim=hp.emb_dim
        self.hidden_size=hp.hidden_size
        self.num_recurrent_layers=hp.recurrent_layers
        self.hp=hp
        assert hp.critic in ['q','v']
        
        if self.lstm_on:
            if lstm == None:
                self.lstm = nn.LSTM(self.emb_dim, self.hidden_size, num_layers=self.num_recurrent_layers)
            else:
                self.lstm = lstm
            if hp.critic == 'q':
                self.theta6_v = nn.Linear(self.hidden_size, self.emb_dim, True)
                self.theta7_v = nn.Linear(self.hidden_size, self.emb_dim, True)        
                self.theta5_v = nn.Linear(2*self.emb_dim, 1, True)
        else:
            if hp.critic == 'q':
                self.theta6_v = nn.Linear(self.emb_dim, self.emb_dim, True)
                self.theta7_v = nn.Linear(self.emb_dim, self.emb_dim, True)
                self.theta5_v = nn.Linear(2*self.emb_dim, 1, True)
        if hp.critic == 'v':
            self.theta8_v = nn.Linear(self.emb_dim, 1, True)
        self.hidden_cell = None

    def reset_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size).to(device),
                            torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size).to(device))

    def forward(self, features, terminal=None, selector=None):
        # features: [seq_len, bsize, feat_dim=emb_dim+3]
        seq_len = features.shape[0]
        max_nodes = self.hp.max_possible_nodes
        assert len(features.shape)==3
        #num_nodes_in_batch = features.shape[1]
        device = features.device
        
        #features_ = features.reshape(seq_len,-1,self.hp.emb_dim+3)
        num_nodes= features.shape[1]
        mu_raw, batch, reachable, num_nodes_list = torch.split(features, [self.emb_dim,1,1,1], dim=2)
        
        batch_size = int(batch.max()) + 1
        batch = batch.squeeze(-1).to(torch.int64)
        reachable = reachable.squeeze(-1).bool()

        if self.hidden_cell is None or max_nodes * batch_size != self.hidden_cell[0].shape[1]:
            self.reset_init_state(max_nodes * batch_size, device)
        if terminal is not None: # reset hidden cell values for nodes part of a terminated graph to 0
            terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(batch_size,dtype=torch.int64)*max_nodes, dim=0)
            assert self.hidden_cell[0][:,terminal_dense,:].sum() == 0
            assert self.hidden_cell[1][:,terminal_dense,:].sum() == 0

        num_nodes_list = num_nodes_list[0].squeeze()

        if self.lstm_on:
            if selector == None:
                selector_=[]
                for i in range(batch_size):
                    selector_ += [True]*int(num_nodes_list[i]) + [False]*(max_nodes-int(num_nodes_list[i]))
                selector=torch.tensor(selector_,dtype=torch.bool)
            full_output, out_hidden_cell  = self.lstm(mu_raw, (self.hidden_cell[0][:,selector,:],self.hidden_cell[1][:,selector,:]) )
            self.hidden_cell[0][:,selector,:] = out_hidden_cell[0]
            self.hidden_cell[1][:,selector,:] = out_hidden_cell[1]
            mu_mem = full_output
        else:
            mu_mem = mu_raw
        mu_mem_meanpool = scatter_mean(mu_mem, batch, dim=1) # (batch_size, emb_dim)
        
        if self.hp.critic == 'q':
            expander = num_nodes_list[:batch_size].to(torch.int64)       
            mu_mem_meanpool_expanded = torch.repeat_interleave(mu_mem_meanpool, expander, dim=1) # (num_nodes in batch, emb_dim)

            # Transform node embeddings to graph state / node values
            global_state = self.theta6_v(mu_mem_meanpool_expanded) # yields (#nodes in batch, emb_dim)
            local_action = self.theta7_v(mu_mem)  # yields (#nodes in batch, emb_dim)
            
            rep = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (#nodes in batch, 2*emb_dim)        
            qvals = self.theta5_v(rep).squeeze(-1)  #(seq_len, nr_nodes in batch)
            qvals[~reachable] = -torch.inf
            v = scatter_max(qvals, batch, dim=1)[0]
        else:
            v = self.theta8_v(mu_mem_meanpool).squeeze(-1)
            

        return v.unsqueeze(-1) # returns (seq_len, batch_size, 1)

class Actor_concat(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev=None, hp=None):
        super().__init__()
        #self.counter = 0
        self.lstm_on = hp.lstm_on
        self.hp = hp
        self.emb_dim = hp.emb_dim
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space 
        self.hidden_size = hp.hidden_size # concat mu and AGG(mu)
        self.num_recurrent_layers = hp.recurrent_layers
        
        if self.lstm_on:
            self.lstm = nn.LSTM(self.emb_dim*2, self.hidden_size * 2, num_layers = self.num_recurrent_layers)
            self.theta6_pi = nn.Linear(self.hidden_size, self.emb_dim, True)
            self.theta7_pi = nn.Linear(self.hidden_size, self.emb_dim, True)
        else:
            self.theta6_pi = nn.Linear(self.emb_dim, self.emb_dim, True)
            self.theta7_pi = nn.Linear(self.emb_dim, self.emb_dim, True)

        self.theta5_pi = nn.Linear(2*self.emb_dim, 1, True)
        self.log_std_dev = nn.Parameter(init_log_std_dev * torch.ones(33, dtype=torch.float), requires_grad=trainable_std_dev)
        self.covariance_eye = torch.eye(33).unsqueeze(0)
        self.hidden_cell = None
        
        #print('Actor network:')
        #self.numTrainableParameters()
        #print(self)
        
    def reset_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size*2).to(device),
                            torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size*2).to(device))
        
    def forward(self, features, terminal=None, selector=None):
        # features: (seq_len, num nodes in batch, feat_dim=emb_dim+3)
        assert len(features.shape)==3
        seq_len = features.shape[0]
        max_nodes = self.hp.max_possible_nodes
        num_nodes= features.shape[1]
        device = features.device

        mu_raw, batch, reachable, num_nodes_list = torch.split(features,[self.emb_dim,1,1,1],dim=2)
        batch_size = int(batch.max()) + 1 
        
        batch = batch.squeeze(-1).to(torch.int64) 
        reachable = reachable.squeeze(-1).bool() 

        if self.hidden_cell is None or max_nodes * batch_size != self.hidden_cell[0].shape[1]:
            self.reset_init_state(max_nodes * batch_size, device)
        if terminal is not None:
            # Set node cells of graph of the termined episode to 0
            terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(batch_size,dtype=torch.int64)*max_nodes, dim=0)
            assert self.hidden_cell[0][:,terminal_dense,:].sum() == 0
            assert self.hidden_cell[1][:,terminal_dense,:].sum() == 0

        num_nodes_list = num_nodes_list[0].squeeze()
        if selector == None:
            selector_=[]
            for i in range(batch_size):
                selector_+=[True]*int(num_nodes_list[i]) + [False]*(max_nodes-int(num_nodes_list[i]))
            selector=torch.tensor(selector_,dtype=torch.bool)   

        mu_raw_meanpool = scatter_mean(mu_raw, batch, dim=1) # (seq_len, batch_size, emb_dim)
        expander = num_nodes_list[:batch_size].to(torch.int64)   
        mu_raw_meanpool_expanded = torch.repeat_interleave(mu_raw_meanpool, expander, dim=1) # (num_nodes in batch, emb_dim)
        
        # Transform node embeddings to action log probabilities
        global_state_raw = self.theta6_pi(mu_raw_meanpool_expanded) # yields (#nodes in batch, emb_dim)
        local_action_raw = self.theta7_pi(mu_raw)  # yields (#nodes in batch, emb_dim)
        
        mu_concat_raw = torch.cat([global_state_raw, local_action_raw], dim=2) # concat creates (#nodes in batch, 2*emb_dim)        

        if self.lstm_on:
            full_output, out_hidden_cell = self.lstm(F.relu(mu_concat_raw), (self.hidden_cell[0][:,selector,:], self.hidden_cell[1][:,selector,:]))
            self.hidden_cell[0][:,selector,:] = out_hidden_cell[0]
            self.hidden_cell[1][:,selector,:] = out_hidden_cell[1]
            mu_concat_mem = full_output # (seq_len, num_nodes in batch, emb_dim)
        else:
            assert False # so sense in concatenated lstm if the lstm is turned off

        #rep = F.relu(mu_concat_mem) # concat creates (#nodes in batch, 2*emb_dim)        
        rep = F.relu(mu_concat_mem)
        prob_logits_raw = self.theta5_pi(rep).squeeze(-1) # (nr_nodes in batch,)

        # Apply action masking and pad to standard size (based on maximum Graph size in nodes in the trainset)
        prob_logits_raw[~reachable] = -torch.inf
        splitter = num_nodes_list[:batch_size].to(torch.int64).tolist()
        if prob_logits_raw.shape[0]==1:
            prob_logits_splitted = list(torch.split(prob_logits_raw.squeeze(), splitter, dim=0))
            p = max_nodes - prob_logits_splitted[0].shape[-1]
            prob_logits_splitted[0] = torch.nn.functional.pad(prob_logits_splitted[0], (0,p), value = -torch.inf)
            prob_logits_splitted = torch.nn.utils.rnn.pad_sequence(prob_logits_splitted, batch_first=True, padding_value = -torch.inf)
            prob_logits=prob_logits_splitted[None,:,:]
        else:
            prob_logits_splitted = list(torch.split(prob_logits_raw.squeeze(), splitter, dim=1))
            for i in range(len(prob_logits_splitted)):
                p = max_nodes - prob_logits_splitted[i].shape[-1]
                prob_logits_splitted[i] = torch.nn.functional.pad(prob_logits_splitted[i], (0,p), value=-torch.inf)
            prob_logits=torch.stack(prob_logits_splitted).permute((1,0,2))
       
        if self.continuous_action_space:
            assert False # not implemented
        else:
            policy_dist = distributions.Categorical(logits=prob_logits.to('cpu'))
        return policy_dist

    def numTrainableParameters(self):
        print('Actor size:')
        print('------------------------------------------')
        total = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("Total number of trainable parameters: {}".format(total))
        print('------------------------------------------')
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

class Critic_concat(nn.Module):
    def __init__(self, state_dim, hp, lstm=None):
        super().__init__()
        self.lstm_on = hp.lstm_on
        self.emb_dim=hp.emb_dim
        self.hidden_size=hp.hidden_size
        self.num_recurrent_layers=hp.recurrent_layers
        self.hp=hp
        #assert hp.critic in ['q'] # v not implemented

        if self.lstm_on:
            if lstm == None:
                self.lstm = nn.LSTM(self.emb_dim*2, self.hidden_size*2, num_layers=self.num_recurrent_layers)
            else:
                self.lstm = lstm
            self.theta6_v = nn.Linear(self.hidden_size, self.emb_dim, True)
            self.theta7_v = nn.Linear(self.hidden_size, self.emb_dim, True)
        else:
            self.theta6_v = nn.Linear(self.emb_dim, self.emb_dim, True)
            self.theta7_v = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta5_v = nn.Linear(2*self.emb_dim, 1, True)
        self.hidden_cell = None
        
    def reset_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size*2).to(device),
                            torch.zeros(self.hp.recurrent_layers, batch_size, self.hidden_size*2).to(device))
    
    def forward(self, features, terminal=None, selector=None):
        # features: [seq_len, bsize, feat_dim=emb_dim+3]
        seq_len = features.shape[0]
        max_nodes = self.hp.max_possible_nodes
        assert len(features.shape)==3
        #num_nodes_in_batch = features.shape[1]
        device = features.device
        
        #features_ = features.reshape(seq_len,-1,self.hp.emb_dim+3)
        num_nodes= features.shape[1]
        mu_raw, batch, reachable, num_nodes_list = torch.split(features, [self.emb_dim,1,1,1], dim=2)
        
        batch_size = int(batch.max()) + 1
        batch = batch.squeeze(-1).to(torch.int64)
        reachable = reachable.squeeze(-1).bool()

        if self.hidden_cell is None or max_nodes * batch_size != self.hidden_cell[0].shape[1]:
            self.reset_init_state(max_nodes * batch_size, device)
        if terminal is not None: # reset hidden cell values for nodes part of a terminated graph to 0
            terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(batch_size,dtype=torch.int64)*max_nodes, dim=0)
            assert self.hidden_cell[0][:,terminal_dense,:].sum() == 0
            assert self.hidden_cell[1][:,terminal_dense,:].sum() == 0

        num_nodes_list = num_nodes_list[0].squeeze()
        if selector == None:
            selector_=[]
            for i in range(batch_size):
                selector_ += [True]*int(num_nodes_list[i]) + [False]*(max_nodes-int(num_nodes_list[i]))
            selector=torch.tensor(selector_,dtype=torch.bool)

        mu_raw_meanpool = scatter_mean(mu_raw, batch, dim=1) # (batch_size, emb_dim)
        

        expander = num_nodes_list[:batch_size].to(torch.int64)       
        mu_raw_meanpool_expanded = torch.repeat_interleave(mu_raw_meanpool, expander, dim=1) # (num_nodes in batch, emb_dim)

        # Transform node embeddings to graph state / node values
        global_state_raw = self.theta6_v(mu_raw_meanpool_expanded) # yields (#nodes in batch, emb_dim)
        local_action_raw = self.theta7_v(mu_raw)  # yields (#nodes in batch, emb_dim)
        mu_concat_raw = torch.cat([global_state_raw, local_action_raw], dim=2)

        if self.lstm_on:
            full_output, out_hidden_cell  = self.lstm(F.relu(mu_concat_raw), (self.hidden_cell[0][:,selector,:],self.hidden_cell[1][:,selector,:]) )
            self.hidden_cell[0][:,selector,:] = out_hidden_cell[0]
            self.hidden_cell[1][:,selector,:] = out_hidden_cell[1]
            mu_concat_mem = full_output
        else:
            assert False # so sense in concatenated lstm if the lstm is turned off

        #rep = F.relu(mu_concat_mem) # concat creates (#nodes in batch, 2*emb_dim)        
        rep=F.relu(mu_concat_mem)
        qvals = self.theta5_v(rep).squeeze(-1)  #(seq_len, nr_nodes in batch)
        qvals[~reachable] = -torch.inf
        v = scatter_max(qvals,batch, dim=1)[0]

        return v.unsqueeze(-1) # returns (seq_len, nr_nodes in batch, 1)
