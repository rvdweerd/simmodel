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

class MaskablePPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev=None, hp=None):
        super().__init__()
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space 
        self.hp=hp

        self.FE = FeatureExtractor(state_dim, hp)
        self.PI = Actor(state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev, hp=hp)
        self.V  = Critic(state_dim,hp)
        print(self)

class FeatureExtractor(nn.Module):
    def __init__(self, state_dim, hp):
        super().__init__()
        self.hp=hp
        self.emb_dim=hp.emb_dim
        self.T=5
        self.node_dim=7
        kwargs={'concat':True}
        self.gat = GATv2(
            in_channels = self.node_dim,
            hidden_channels = self.emb_dim,
            heads = 1,
            num_layers = self.T,
            out_channels = self.emb_dim,
            share_weights = False,
            **kwargs
        )
        layers = [nn.Linear(state_dim,state_dim), nn.ELU()]
        self.lin_layers = nn.Sequential(*layers)

    def deserialize(self, obs):
        num_nodes, max_nodes, num_edges, max_edges, F = obs[-5:].to(torch.int64).tolist()
        assert num_nodes==max_nodes
        nf,py,re,_ = torch.split(obs,(F*max_nodes, 2*max_edges, max_nodes, 5),dim=0)
        nf=nf.reshape(max_nodes,-1)#[:num_nodes]
        py=py.reshape(2,-1)[:,:num_edges].to(torch.int64)
        #re=re.reshape(-1,1)[:num_nodes].to(torch.int64)
        re=re.reshape(-1,1).to(torch.int64)
        return nf,py,re,num_nodes,max_nodes,num_edges,max_edges,F
        
    def forward(self, state):
        # state: (bize,flatvecdim)
        bsize=state.shape[0]
        pyg_list=[]
        #num_nodes_list=[]
        reachable_nodes_tensor=torch.tensor([])
        for i in range(bsize):
            pygx,pygei,reachable_nodes,num_nodes,max_nodes,num_edges,max_edges,F = self.deserialize(state[i])
            pyg_list.append(Data(
                pygx,
                pygei
            ))
            #num_nodes_list.append(num_nodes)
            reachable_nodes_tensor = torch.cat((reachable_nodes_tensor,reachable_nodes.squeeze()))
        pyg_data = Batch.from_data_list(pyg_list)
        mu_raw = self.gat(pyg_data.x, pyg_data.edge_index)  	# (num_nodes_in_batch,emb_dim)
        #mu_meanpool = scatter_mean(mu_raw,pyg_data.batch,dim=0) # (bsize,emb_dim)
        #num_nodes_list += [0]*(pyg_data.num_nodes- len(num_nodes_list))
        #num_nodes_tensor = torch.tensor(num_nodes_list)
        features = torch.cat((mu_raw, pyg_data.batch[:,None], reachable_nodes_tensor[:,None]),dim=1)
        return features.reshape(bsize,-1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev=None, hp=None):
        super().__init__()
        self.counter=0
        self.hp=hp
        self.emb_dim=hp.emb_dim
        self.hidden_size=hp.hidden_size
        self.num_recurrent_layers=hp.recurrent_layers
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_size, num_layers=self.num_recurrent_layers)
        self.theta5_pi = nn.Linear(2*self.emb_dim, 1, True)#, dtype=torch.float32)
        self.theta6_pi = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta7_pi = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)

        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space 
        self.log_std_dev = nn.Parameter(init_log_std_dev * torch.ones((action_dim), dtype=torch.float), requires_grad=trainable_std_dev)
        self.covariance_eye = torch.eye(self.action_dim).unsqueeze(0)
        self.hidden_cell = None
        print('Actor network:')
        self.numTrainableParameters()
        print(self)
        
    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.hp.recurrent_layers, batch_size, self.hp.hidden_size).to(device),
                            torch.zeros(self.hp.recurrent_layers, batch_size, self.hp.hidden_size).to(device))
        
    def forward(self, features, terminal=None, batch_data=None):
        batch_size = len(terminal)#state.shape[1]
        num_nodes_in_batch = features.shape[1]
        device = features.device
        mu_raw, batch, reachable, num_nodes = torch.split(features,[self.emb_dim,1,1,1],dim=2)
        num_nodes = num_nodes.squeeze()[:batch_size].to(torch.int64)
        batch = batch.squeeze().to(torch.int64)
        #terminal = torch.repeat_interleave(terminal, num_nodes, dim=0)
        terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(batch_size,dtype=torch.int64)*self.hp.max_possible_nodes, dim=0)

        if self.hidden_cell is None or self.hp.max_possible_nodes * batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(self.hp.max_possible_nodes * batch_size, device)
        if terminal is not None:
            self.hidden_cell[0][:,terminal_dense,:] = 0.
            self.hidden_cell[1][:,terminal_dense,:] = 0.
        self.counter+=1

        active_hidden_rows = batch_data[:,2].to(torch.bool) # node select indicator
        _, hidden_update = self.lstm(mu_raw, (self.hidden_cell[0][:,active_hidden_rows,:],self.hidden_cell[1][:,active_hidden_rows,:]))
        self.hidden_cell[0][:,active_hidden_rows,:] = hidden_update[0]
        self.hidden_cell[1][:,active_hidden_rows,:] = hidden_update[1]
        
        assert (batch_data[:,0][active_hidden_rows].to(torch.int64) == batch).all()      
        mu_mem = self.hidden_cell[0][-1][active_hidden_rows] # (num_nodes in batch, emb_dim)
        
        mu_mem_meanpool = scatter_mean(mu_mem, batch.to(torch.int64), dim=0) # (batch_size, emb_dim)
        mu_mem_meanpool_expanded = torch.repeat_interleave(mu_mem_meanpool, num_nodes, dim=0) # (num_nodes in batch, emb_dim)
        
        global_state = self.theta6_pi(mu_mem_meanpool_expanded) # yields (#nodes in batch, emb_dim)
        local_action = self.theta7_pi(mu_mem)  # yields (#nodes in batch, emb_dim)
        rep = F.relu(torch.cat([global_state, local_action], dim=1)) # concat creates (#nodes in batch, 2*emb_dim)        

        prob_logits_raw = self.theta5_pi(rep).squeeze(-1) # (nr_nodes in batch,)

        splitter = features[0][:batch_size,-1].to(torch.int64).tolist()
        reachable=reachable.squeeze().to(torch.bool)
        
        # METHOD 1: SPLITTER (TODO check grad_fn; is gradient retained after split and list operations)
        if features.shape[0]>1:
            splitter1 = features[1][:batch_size,-1].to(torch.int64).tolist()
            assert (splitter==splitter1).all()
        #prob_logits_raw_masked = prob_logits_raw.clone()
        prob_logits_raw[~reachable]=-torch.inf
        splitted_logits = list(prob_logits_raw.split(splitter,dim=0))
        p = self.hp.max_possible_nodes-splitter[0]
        assert p>=0
        splitted_logits[0] = torch.nn.functional.pad(splitted_logits[0],(0,p),value=-torch.inf)
        prob_logits=torch.nn.utils.rnn.pad_sequence(splitted_logits,batch_first=True,padding_value=-torch.inf)        

        # METHOD 2: FOR LOOP
        # prob_logits=torch.ones(batch_size, self.hp.max_possible_nodes).to(torch.float32).to(device)*(-torch.inf) # (batch_size, max_num_nodes)
        # start=0
        # for i in range(batch_size):
        #     reach = reachable[start:(start+int(num_nodes[i]))]#.to(torch.bool)
        #     prob_logits[i][:int(num_nodes[i])][reach] = prob_logits_raw[start:(start+int(num_nodes[i]))][reach]
        #     start+=int(num_nodes[i])
       
        if self.continuous_action_space:
            assert False
            # cov_matrix = self.covariance_eye.to(device).expand(batch_size, self.action_dim, self.action_dim) * torch.exp(self.log_std_dev.to(device))
            # # We define the distribution on the CPU since otherwise operations fail with CUDA illegal memory access error.
            # policy_dist = torch.distributions.multivariate_normal.MultivariateNormal(policy_logits_out.to("cpu"), cov_matrix.to("cpu"))
        else:
            #policy_dist = distributions.Categorical(F.softmax(prob_logits, dim=1).to("cpu"))
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
    def __init__(self, state_dim, hp):
        super().__init__()
        self.linsizes=[64,64]
        self.emb_dim=hp.emb_dim
        self.hidden_size=hp.hidden_size
        self.num_recurrent_layers=hp.recurrent_layers
        self.hp=hp
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_size, num_layers=self.num_recurrent_layers)
        self.theta5_v = nn.Linear(2*self.emb_dim, 1, True)#, dtype=torch.float32)
        self.theta6_v = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta7_v = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)

        self.counter=0
        if type(self.linsizes) is not list:
            assert False
        layers=[]
        layer_sizes=[hp.hidden_size]+self.linsizes
        for layer_idx in range(1,len(layer_sizes)):
            layers+= [nn.Linear(layer_sizes[layer_idx-1], layer_sizes[layer_idx]), nn.ELU() ]
        layers+= [nn.Linear(layer_sizes[-1], 1)]
        self.lin_layers = nn.Sequential(*layers)
        self.hidden_cell = None
        
    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.hp.recurrent_layers, batch_size, self.hp.hidden_size).to(device),
                            torch.zeros(self.hp.recurrent_layers, batch_size, self.hp.hidden_size).to(device))
    
    def forward(self, features, terminal=None, batch_data=None):
        batch_size = len(terminal)#state.shape[1]
        seq_len = features.shape[0]
        max_nodes = self.hp.max_possible_nodes
        assert batch_size==features.shape[1]
        assert len(features.shape)==3
        #num_nodes_in_batch = features.shape[1]
        device = features.device
        
        features_ = features.reshape(seq_len,batch_size*max_nodes,-1)
        mu_raw, batch, reachable = torch.split(features_,[self.emb_dim,1,1],dim=2)
        #num_nodes = num_nodes.squeeze()[:batch_size].to(torch.int64)
        batch = batch.squeeze().to(torch.int64)
        #terminal_sparse = torch.repeat_interleave(terminal.to(torch.int64), num_nodes, dim=0)
        terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(batch_size,dtype=torch.int64)*max_nodes, dim=0)
        if self.hidden_cell is None or max_nodes * batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(max_nodes * batch_size, device)
        if terminal is not None: # reset hidden cell values for nodes part of a terminated graph to 0
            self.hidden_cell[0][:,terminal_dense,:] = 0.
            self.hidden_cell[1][:,terminal_dense,:] = 0.
        self.counter+=1

        #active_hidden_rows = batch_data[:,2].to(torch.bool) # node select indicator
        #_, self.hidden_cell = self.lstm(mu_raw, self.hidden_cell)
        _, self.hidden_cell  = self.lstm(mu_raw, self.hidden_cell)
        #self.hidden_cell[0][:,active_hidden_rows,:] = hidden_update[0]
        #self.hidden_cell[1][:,active_hidden_rows,:] = hidden_update[1]
        #value_out = self.lin_layers(self.hidden_cell[0][-1]) # COULD APPLY LINEAR LAYERS + NONLINs HERE
        #assert (batch_data[:,0][active_hidden_rows].to(torch.int64) == batch).all()
        mu_mem = self.hidden_cell[0][-1] # (num_nodes in batch, emb_dim)

        mu_mem_meanpool = scatter_mean(mu_mem, batch.to(torch.int64), dim=0) # (batch_size, emb_dim)
        mu_mem_meanpool_expanded = torch.repeat_interleave(mu_mem_meanpool, torch.ones(batch_size,dtype=torch.int64)*max_nodes, dim=0) # (num_nodes in batch, emb_dim)
        
        global_state = self.theta6_v(mu_mem_meanpool_expanded) # yields (#nodes in batch, emb_dim)
        local_action = self.theta7_v(mu_mem)  # yields (#nodes in batch, emb_dim)
        rep = F.relu(torch.cat([global_state, local_action], dim=1)) # concat creates (#nodes in batch, 2*emb_dim)        
        qvals = self.theta5_v(rep).squeeze(-1)  #(nr_nodes in batch,)
        qvals[~reachable.squeeze().bool()] = -torch.inf
        v=scatter_max(qvals,batch,dim=0)[0]

        return v.unsqueeze(-1) #(bsize,1)

def encode_hidden(hidden_cell, enc_info):
    # padds and packs hidden cell to standardized shape
    # hidden_cell: (num_nodes in batch, emb_dim)
    complement = torch.cat((hidden_cell,enc_info['batch_data']),dim=1)
    return complement

def encode_features(features, enc_info):
    splitter = features[:enc_info['batch_size'],-1].to(torch.int64).tolist()
    features[:,-1]=1
    splitted_features = list(features.split(splitter,dim=0))
    p = enc_info['Vmax']-splitter[0]
    splitted_features[0]=torch.nn.functional.pad(splitted_features[0],(0,0,0,p))
    features=torch.nn.utils.rnn.pad_sequence(splitted_features,batch_first=True)
    return features.reshape(enc_info['batch_size']*enc_info['Vmax'], -1) # (max_nodes_in_batch, emb_dim+3)