import numpy as np
import torch
import random
import math
from collections import namedtuple
import os
import time
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
#from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import medfilt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class QNet(nn.Module):
    """ The neural net that will parameterize the function Q(s, a)
    
        The input is the state (containing the graph and visited nodes),
        and the output is a vector of size N containing Q(s, a) for each of the N actions a.
    """    
    
    def __init__(self, config):
        """ emb_dim: embedding dimension p
            T: number of iterations for the graph embedding
        """
        super(QNet, self).__init__()
        self.emb_dim = config['emb_dim']
        self.T = config['emb_iter_T']
        
        # We use 5 dimensions for representing the nodes' states:
        # * A binary variable indicating whether the node has been visited
        # * A binary variable indicating whether the node is the first of the visited sequence
        # * A binary variable indicating whether the node is the last of the visited sequence
        # * The node number //The (x, y) coordinates of the node.
        self.node_dim = config['node_dim']
        
        # We can have an extra layer after theta_1 (for the sake of example to make the network deeper)
        #nr_extra_layers_1 = config['num_extra_layers']
        
        # Build the learnable affine maps:
        self.theta1a = nn.Linear(self.node_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta1b = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        #torch.nn.init.xavier_normal_(self.theta1.weight)
        self.theta2 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta3 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta4 = nn.Linear(1, self.emb_dim, True, dtype=torch.float32)
        self.theta5 = nn.Linear(2*self.emb_dim, 1, True, dtype=torch.float32)
        self.theta6 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta7 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        
        #self.theta1_extras = [nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32).to(device) for _ in range(nr_extra_layers_1)]
        self.numTrainableParameters()

    def forward(self, xv, Ws):
        # xv: The node features (batch_size, num_nodes, node_dim)
        # Ws: The graphs (batch_size, num_nodes, num_nodes)
        
        num_nodes = xv.shape[1]
        batch_size = xv.shape[0]
        
        # pre-compute 1-0 connection matrices masks (batch_size, num_nodes, num_nodes)
        conn_matrices = torch.where(Ws > 0, torch.ones_like(Ws), torch.zeros_like(Ws)).to(device)
        
        # Graph embedding
        # Note: we first compute s1 and s3 once, as they are not dependent on mu
        mu = torch.zeros(batch_size, num_nodes, self.emb_dim, device=device, dtype=torch.float32)
        #s1 = self.theta1a(xv)  # (batch_size, num_nodes, emb_dim)
        s1 = self.theta1b(F.relu(self.theta1a(xv)))  # (batch_size, num_nodes, emb_dim)
        #for layer in self.theta1_extras:
        #    s1 = layer(F.relu(s1))  # we apply the extra layer
        
        s3_1 = F.relu(self.theta4(Ws.unsqueeze(3)))  # (batch_size, nr_nodes, nr_nodes, emb_dim) - each "weigth" is a p-dim vector        
        s3_2 = torch.sum(s3_1, dim=1)  # (batch_size, nr_nodes, emb_dim) - the embedding for each node
        s3 = self.theta3(s3_2)  # (batch_size, nr_nodes, emb_dim)
        
        for t in range(self.T):
            s2 = self.theta2(conn_matrices.matmul(mu))    
            mu = F.relu(s1 + s2 + s3)
            
        """ prediction
        """
        # we repeat the global state (summed over nodes) for each node, 
        # in order to concatenate it to local states later
        global_state = self.theta6(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1))
        
        local_action = self.theta7(mu)  # (batch_dim, nr_nodes, emb_dim)
            
        out = F.relu(torch.cat([global_state, local_action], dim=2))
        return self.theta5(out).squeeze(dim=2)

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


class QFunction():
    def __init__(self, model, optimizer, lr_scheduler):
        self.model = model  # The actual QNet
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        #self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()
    
    def predict(self, state_tsr, W):
        # batch of 1 - only called at inference time
        with torch.no_grad():
            estimated_rewards = self.model(state_tsr.unsqueeze(0), W.unsqueeze(0))
        return estimated_rewards[0]
                
    def get_best_action(self, state_tsr, W, reachable_nodes):
        """ Computes the best (greedy) action to take from a given state
            Returns a tuple containing the ID of the next node and the corresponding estimated reward
        """
        W = torch.tensor(W,dtype=torch.float32,device=device)
        estimated_rewards = self.predict(state_tsr, W)  # size (nr_nodes,)
        reachable_rewards = estimated_rewards[reachable_nodes]
        #sorted_reward_idx = estimated_rewards.argsort(descending=True)
        #print(reachable_rewards)
        #print(state_tsr)
        #print(W)
        action = torch.argmax(reachable_rewards).detach().item()
        action_nodeselect = reachable_nodes[action]
        best_reward = reachable_rewards[action]
        return action, action_nodeselect, best_reward
        
        
    def batch_update(self, states_tsrs, Ws, actions, targets):
        """ Take a gradient step using the loss computed on a batch of (states, Ws, actions, targets)
        
            states_tsrs: list of (single) state tensors
            Ws: list of W tensors
            actions: list of actions taken
            targets: list of targets (resulting estimated rewards after taking the actions)
        """        
        Ws_tsr = torch.stack(Ws).to(device)
        xv = torch.stack(states_tsrs).to(device)
        self.optimizer.zero_grad()
        
        # the rewards estimated by Q for the given actions
        estimated_rewards = self.model(xv, Ws_tsr)[range(len(actions)), actions]
        
        loss = self.loss_fn(estimated_rewards, torch.tensor(targets, device=device))
        loss_val = loss.item()
        # check grads: self.model.theta4.weight.grad
        loss.backward()
        self.optimizer.step()        
        self.lr_scheduler.step()
        
        return loss_val

def init_model(config, fname=None):
    """ Create a new model. If fname is defined, load the model from the specified file.
    """
    Q_net = QNet(config).to(device)
    optimizer = optim.Adam(Q_net.parameters(), config['lr_init'])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])
    
    if fname is not None:
        checkpoint = torch.load(fname)
        Q_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    Q_func = QFunction(Q_net, optimizer, lr_scheduler)
    return Q_func, Q_net, optimizer, lr_scheduler

def checkpoint_model(model, optimizer, lr_scheduler, loss, 
                     episode, avg_length, logdir='test', best_only=False):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        
    fname_impr = os.path.join(logdir, 'ep_{}'.format(episode))
    fname_impr += '_length_{}'.format(avg_length)
    fname_impr += '.tar'
    fname_best = os.path.join(logdir, 'best_model.tar')
    fnames=[fname_best]
    print(fname_impr)
    if not best_only:
        pass#fnames.append(fname_impr)
    for fname in fnames:
        torch.save({
            'episode': episode,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'loss': loss,
            'avg_length': avg_length
        }, fname)

class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.nr_inserts = 0
        
    def remember(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        self.nr_inserts += 1
        
    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return min(self.nr_inserts, self.capacity)