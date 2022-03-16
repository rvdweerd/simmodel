import numpy as np
import torch
import tqdm
import random
import math
from collections import namedtuple
import os
import time
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from modules.dqn.dqn_utils import seed_everything
from modules.rl.rl_policy import GNN_s2v_Policy, ShortestPathPolicy, EpsilonGreedyPolicy
from modules.rl.rl_algorithms import q_learning_exhaustive
from modules.rl.rl_utils import EvaluatePolicy, EvalArgs1, EvalArgs2, EvalArgs3, GetFullCoverageSample
from modules.sim.simdata_utils import SimulateInteractiveMode
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import GAT
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_scatter import scatter_mean, scatter_add
from scipy.signal import medfilt
import copy
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
Experience = namedtuple('Experience', ( \
    'state', 
    'state_tsr',# node feature matrix (nfm)
    'W',        # weight/adjaceny matrix
    'pyg_data', # Pytorch Geometric data object (nfm+edge_index)
    'action', 
    'action_nodeselect', 
    'reward', 
    'done', 
    'next_state', 
    'next_state_tsr', 
    'next_pyg_data',
    'next_state_neighbors'))


device = 'cuda' if torch.cuda.is_available() else 'cpu'


from torch_geometric.nn.conv import MessagePassing, GATv2Conv
from torch_geometric.nn.models.basic_gnn import BasicGNN
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

class QNet_GAT(nn.Module):
    """ Graph Attention based Q function
    """    
    
    def __init__(self, config):
        """ emb_dim: embedding dimension p
            T: number of iterations for the graph embedding
        """
        super(QNet_GAT, self).__init__()
        self.emb_dim    = config['emb_dim']
        self.num_layers = config['emb_iter_T']
        self.node_dim   = config['node_dim']
        self.norm_agg = config['norm_agg']
        kwargs={'concat':True}
        self.gat = GAT(
            in_channels = self.node_dim,
            hidden_channels = self.emb_dim,
            num_layers = 5,
            out_channels = self.emb_dim,
            **kwargs
        ).to(device)
        self.theta5 = nn.Linear(2*self.emb_dim, 1, True, dtype=torch.float32)
        self.theta6 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta7 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)     

        self.numTrainableParameters()

    def forward(self, xv, Ws, pyg_data):
        # xv: The node features (batch_size, num_nodes, node_dim)
        # Ws: The graphs (batch_size, num_nodes, num_nodes)
        # pyg_data: pytorch geometric Batch
        #num_nodes_padded = xv.shape[1]
        
        #test=self.gat(pyg_data.x, pyg_data.edge_index).shape # gives (sum of num_nodes in batch, emb_dim)
        # self.gat is an instantiated torch_geometric.nn.models.GAT object
        ptr = pyg_data.ptr.detach().cpu().numpy()
        #splitter = [r-l for r,l in zip(ptr[1:], ptr[:-1])] # list of #nodes of each graph in the batch
        splitter_STAR = list(ptr[1:]-ptr[:-1])
        mu_raw = self.gat(pyg_data.x ,pyg_data.edge_index) # yields the node embeddings (sum of num_nodes in the batch, emb_dim)
        # mu_raw has shape (sum of num_nodes in the batch, emb_dim), mu needs to be unpacked
        # mu = torch.nn.utils.rnn.pad_sequence(mu_raw.split(splitter_STAR, dim=0), batch_first=True) #(bsize,max_num_nodes,emb_dim)
        # if mu.shape[1] < num_nodes_padded:
        #     p = num_nodes_padded - mu.shape[1]
        #     mu = nn.functional.pad(mu, (0,0,0,p,0,0))

        """ prediction
        """

        # we repeat the global state (summed over nodes) for each node, 
        # in order to concatenate it to local states later
        if self.norm_agg:
            mu_meanpool = scatter_mean(mu_raw,pyg_data.batch,dim=0) # (bsize,emb_dim)
            mu_meanpool_expanded = torch.repeat_interleave(mu_meanpool,torch.tensor(splitter_STAR,dtype=torch.int64,device=device),dim=0) #(sum of num_nodes in the batch, emb_dim)
            #ptr = pyg_data.ptr.detach().cpu().numpy()
            #s = [1/(r-l) for r,l in zip(ptr[1:], ptr[:-1])] # list of #nodes of each graph in the batch
            #nodesizes=torch.tensor(s,dtype=torch.float32,device=device)[:,None,None] #(bsize,1,1)
            #agg=torch.sum(mu, dim=1, keepdim=True) # (bsize,1,emb_dim)
            #normagg=agg*nodesizes # normalized graph representations, (bsize,1,emb_dim)
            #global_state = self.theta6(normagg.repeat(1, num_nodes_padded, 1)) # (bsize,max_num_nodes,emb_dim)
            global_state_STAR = self.theta6(mu_meanpool_expanded)
            #global_state = self.theta6((torch.sum(mu, dim=1, keepdim=True)/pyg_data.num_nodes).repeat(1, num_nodes_padded, 1))
        else:
            mu_maxpool = scatter_add(mu_raw,pyg_data.batch,dim=0) # (bsize,emb_dim)
            mu_maxpool_expanded = torch.repeat_interleave(mu_maxpool,torch.tensor(splitter_STAR,dtype=torch.int64,device=device),dim=0) #(sum of num_nodes in the batch, emb_dim)
            #global_state = self.theta6(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes_padded, 1))
            global_state_STAR = self.theta6(mu_maxpool_expanded)
        
        #local_action = self.theta7(mu)  # (batch_dim, nr_nodes, emb_dim)
        local_action_STAR = self.theta7(mu_raw)  # (sum of num_nodes in the batch, emb_dim)

        out_STAR = F.relu(torch.cat([global_state_STAR, local_action_STAR], dim=1)) # (sum of num_nodes in the batch, 2*emb_dim)
        out_STAR = self.theta5(out_STAR).squeeze() # (sum of num_nodes in the batch)

        #out = F.relu(torch.cat([global_state, local_action], dim=2)) # (bsize, max_num_nodes, 2*emb_dim)
        #out = self.theta5(out).squeeze(dim=2) # (bsize, max_num_nodes)
        return out_STAR

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

class QNet_GATv2(QNet_GAT):
    def __init__(self, config):
        """ emb_dim: embedding dimension p
            T: number of iterations for the graph embedding
        """
        super(QNet_GATv2, self).__init__(config)
        self.emb_dim    = config['emb_dim']
        self.num_layers = config['emb_iter_T']
        self.node_dim   = config['node_dim']
        self.norm_agg = config['norm_agg']
        kwargs={'concat':True}
        self.gat = GATv2(
            in_channels = self.node_dim,
            hidden_channels = self.emb_dim,
            num_layers = 5,
            out_channels = self.emb_dim,
            **kwargs
        ).to(device)
        self.theta5 = nn.Linear(2*self.emb_dim, 1, True, dtype=torch.float32)
        self.theta6 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta7 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)     

        self.numTrainableParameters()

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
        self.norm_agg=config['norm_agg']

        # Dimensions for representing the nodes' states:
        self.node_dim = config['node_dim']      
        
        # Build the learnable affine maps:
        self.theta1a = nn.Linear(self.node_dim, self.emb_dim, True, dtype=torch.float32)#.to(device)
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

    def forward(self, xv, Ws, pyg_data):
        # xv: The node features (batch_size, num_nodes, node_dim)
        # Ws: The graphs (batch_size, num_nodes, num_nodes)
        # pyg_data: pytorch geometric Batch (not used here)
        num_nodes_padded = xv.shape[1]
        #num_nodes_orig = pyg_data.num_nodes
        batch_size = xv.shape[0]
        
        # pre-compute 1-0 connection matrices masks (batch_size, num_nodes, num_nodes)
        conn_matrices = Ws#torch.where(Ws > 0, torch.ones_like(Ws), torch.zeros_like(Ws)).to(device)
        
        # Graph embedding
        # Note: we first compute s1 and s3 once, as they are not dependent on mu
        mu = torch.zeros(batch_size, num_nodes_padded, self.emb_dim, dtype=torch.float32,device=device)
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
        ptr = pyg_data.ptr.detach().cpu().numpy()
        n = [(r-l) for r,l in zip(ptr[1:], ptr[:-1])] # list of #nodes of each graph in the batch
        lst = []
        for i in range(len(n)):
            lst+=[i]*n[i] + [len(n)]*(num_nodes_padded-n[i])
        lst=torch.tensor(lst,dtype=torch.int64,device=device)
        if self.norm_agg:
            normagg = scatter_mean(mu.reshape(-1,self.emb_dim), lst, dim=0)[:-1]
        else:
            normagg = scatter_add(mu.reshape(-1,self.emb_dim), lst, dim=0)[:-1]
        global_state = self.theta6(normagg.repeat(1, num_nodes_padded, 1)) # (bsize,max_num_nodes,emb_dim)
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
    
    def predict(self, state_tsr, W, pyg_data):
        # batch of 1 - only called at inference time
        pyg_batch = Batch.from_data_list([pyg_data]).to(device)
        with torch.no_grad():
            estimated_rewards = self.model(state_tsr.unsqueeze(0), W.unsqueeze(0), pyg_batch)
        if len(estimated_rewards.shape) == 2 and estimated_rewards.shape[0]==1: # bsize is 1
            return estimated_rewards[0]
        elif len(estimated_rewards.shape) == 1:
            return estimated_rewards
        else:
            assert False
                
    def get_best_action(self, state_tsr, W, pyg_data, reachable_nodes, printing=False):
        """ Computes the best (greedy) action to take from a given state
            Returns a tuple containing the ID of the next node and the corresponding estimated reward
        """
        #W = torch.tensor(W,dtype=torch.float32,device=device)
        state_tsr=state_tsr.to(device)
        W=W.to(device)
        self.model.eval()
        estimated_rewards = self.predict(state_tsr, W, pyg_data)  # size (nr_nodes,)
        self.model.train()
        reachable_rewards = estimated_rewards.detach().cpu().numpy()[reachable_nodes]
        #sorted_reward_idx = estimated_rewards.argsort(descending=True)
        #print(reachable_rewards)
        #print(state_tsr)
        #print(W)
        action = np.argmax(reachable_rewards)
        action_nodeselect = reachable_nodes[action]
        best_reward = reachable_rewards[action]
        if printing:
            print('reachable_nodes:',reachable_nodes)
            print('values         :',reachable_rewards)
            print('Selected action:',action,'node',action_nodeselect,'value:',best_reward)
        return action, action_nodeselect, best_reward
        
        
    def batch_update(self, states_tsrs, Ws, pyg_data, actions, targets):
        """ Take a gradient step using the loss computed on a batch of (states, Ws, actions, targets)
        
            states_tsrs: list of (single) state tensors
            Ws: list of W tensors
            actions: list of actions taken
            targets: list of targets (resulting estimated rewards after taking the actions)
        """        
        Ws_tsr = torch.stack(Ws).to(device)
        xv = torch.stack(states_tsrs).to(device)
        pyg_batch = Batch.from_data_list(pyg_data).to(device)
        self.optimizer.zero_grad()
        
        # the rewards estimated by Q for the given actions
        #estimated_rewards = self.model(xv, Ws_tsr)[range(len(actions)), actions]
        qvals_STAR = self.model(xv, Ws_tsr, pyg_batch) # (bsize, max_num_nodes)
        #actionselect = torch.tensor(actions,dtype=torch.int64,device=device)[:,None] #(bsize,1)
        #estimated_rewards = torch.gather(qvals, 1, actionselect).squeeze() #(bsize)
        #estimated_rewards = torch.gather(self.model(xv, Ws_tsr, pyg_batch), 1, torch.tensor(actions,dtype=torch.int64,device=device)[:,None]).squeeze()


        actionselect_STAR = torch.tensor(actions,dtype=torch.int64,device=device) + pyg_batch.ptr[:-1]
        estimated_rewards_STAR = qvals_STAR[actionselect_STAR]

        loss = self.loss_fn(estimated_rewards_STAR, torch.tensor(targets, device=device))
        loss_val = loss.detach().cpu().item()
        # check grads: self.model.theta4.weight.grad
        loss.backward()
        self.optimizer.step()        
        self.lr_scheduler.step()
        
        return loss_val

def init_model(config, fname=None):
    """ Create a new model. If fname is defined, load the model from the specified file.
    """
    if config['qnet'] == 's2v':
        Q_net = QNet(config).to(device)
    elif config['qnet'] == 'gat':
        Q_net = QNet_GAT(config).to(device)
    elif config['qnet'] == 'gat2':
        Q_net = QNet_GATv2(config).to(device)
    else:
        assert False
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


def train(seed=0, config=None, env_all=None):
    # Storing metrics about training:
    found_solutions = dict()  # episode --> (W, solution)
    losses = []
    path_length_ratios = []
    total_rewards=[]
    grad_update_count=0
    zerovec=torch.tensor([0],dtype=torch.float32)
    #zerodat=Data(x=zerovec,edge_index=[0,0])

    seed_everything(seed) # 
    # Create module, optimizer, LR scheduler, and Q-function
    Q_func, Q_net, optimizer, lr_scheduler = init_model(config)
    Q_func_target, _, _, _ = init_model(config)
    #Q_func_target=Q_func

    logdir=config['logdir']+'/SEED'+str(seed)
    writer=writer = SummaryWriter(log_dir=logdir)

    # Create memory
    memory = Memory(config['memory_size'])

    # keep track of mean ratio of estimated MVC / real MVC
    current_min_Ratio = float('+inf')
    current_max_Return= float('-inf')
    N_STEP_QL = config['num_step_ql']
    GAMMA=config['gamma']
    GAMMA_ARR = np.array([GAMMA**i for i in range(N_STEP_QL)])
    
    for episode in range(config['num_episodes']):
        # sample a new graph
        # current state (tuple and tensor)
        env=random.choice(env_all)
        #env=env_all[-1]
        env.reset()
        current_state = env.state
        done=False   
        current_state_tsr = env.nfm.clone() if config['qnet']=='s2v' else zerovec
        current_pyg_data = Data(x=env.nfm.clone(), edge_index=env.sp.EI.clone()) if 'gat' in config['qnet'] else Data(torch.ones(env.sp.V),[0,0])
        # Note: nfm = Graph Feature Matrix (FxV), columns are the node features, managed by the environment
        # It's currently defined (for each node) as:
        #   [.] node number
        #   [.] 1 if target node, 0 otherwise 
        #   [.] # of units present at node at current time
        
        # Padding size to ensure W always same shape to enable batching
        Psize = config['max_nodes'] - env.sp.V if config['qnet']=='s2v' else 0
        assert env.sp.V == env.sp.W.shape[0]

        # Keep track of some variables for insertion in replay memory:
        states = [current_state]
        states_tsrs = [current_state_tsr]  # we also keep the state tensors here (for efficiency)
        pyg_objects = [current_pyg_data]
        rewards = []
        dones = []
        actions = []
        actions_nodeselect = []

        # current value of epsilon
        epsilon = max(config['eps_min'], config['eps_0']*((1-config['eps_decay'])**episode))
        
        nr_explores = 0
        t = -1
        while not done:
            t += 1  # time step of this episode
            
            if epsilon >= random.random():
                # explore
                action = random.randint(0,env.out_degree[env.state[0]]-1)
                #print(env.neighbors[env.state[0]],action)
                action_nodeselect = env.neighbors[env.state[0]][action]
                nr_explores += 1
                if episode % 50 == 0:
                    pass
                    #print('Ep {} explore | current sol: {} | sol: {}'.format(episode, solution, solutions),'nextnode',next_node)
            else:
                # exploit
                #with torch.no_grad(): #(already dealt with inside function)
                reachable_nodes=env.neighbors[env.state[0]]
                #pyg_data = Data(x=env.nfm, edge_index=env.sp.EI)
                action, action_nodeselect, _ = Q_func.get_best_action(current_state_tsr, env.sp.W, current_pyg_data, reachable_nodes)
                if episode % 50 == 0:
                    pass
                    #print('Ep {} exploit | current sol: {} / next est reward: {} | sol: {}'.format(episode, solution, est_reward,solutions),'nextnode',next_node)
            
            _, reward, done, info = env.step(action)
            next_state = env.state
            next_state_tsr = env.nfm.clone() if config['qnet']=='s2v' else zerovec
            next_pyg_data = Data(x=env.nfm.clone(), edge_index=env.sp.EI.clone()) if 'gat' in config['qnet'] else Data(torch.ones(env.sp.V),[0,0])

            # store rewards and states obtained along this episode:
            states.append(next_state)
            states_tsrs.append(next_state_tsr)
            pyg_objects.append(next_pyg_data)
            rewards.append(reward)
            dones.append(done)
            actions.append(action)
            actions_nodeselect.append(action_nodeselect)
            

            # store our experience in memory, using n-step Q-learning:
            if len(actions) >= N_STEP_QL:
                memory.remember(Experience( state          = states[-(N_STEP_QL+1)],
                                            state_tsr      = nn.functional.pad(states_tsrs[-(N_STEP_QL+1)],(0,0,0,Psize)) if config['qnet']=='s2v' else zerovec,
                                            W              = nn.functional.pad(env.sp.W,(0,Psize,0,Psize)) if config['qnet']=='s2v' else zerovec,
                                            pyg_data       = pyg_objects[-(N_STEP_QL+1)], #if 'gat' in config['qnet'] else zerodat,
                                            action         = actions[-N_STEP_QL],
                                            action_nodeselect=actions_nodeselect[-N_STEP_QL],
                                            done           = dones[-1], 
                                            reward         = sum(GAMMA_ARR * np.array(rewards[-N_STEP_QL:])),
                                            next_state     = next_state,
                                            next_state_tsr = nn.functional.pad(next_state_tsr,(0,0,0,Psize)) if config['qnet']=='s2v' else zerovec,
                                            next_pyg_data  = next_pyg_data,
                                            next_state_neighbors= env.neighbors[next_state[0]] ))

            if done:
                for n in range(1, min(N_STEP_QL, len(states))):
                    memory.remember(Experience( state       = states[-(n+1)],
                                                state_tsr   = nn.functional.pad(states_tsrs[-(n+1)],(0,0,0,Psize)) if config['qnet']=='s2v' else zerovec,
                                                W           = nn.functional.pad(env.sp.W,(0,Psize,0,Psize)) if config['qnet']=='s2v' else zerovec,
                                                pyg_data    = pyg_objects[-(n+1)], #if 'gat' in config['qnet'] else zerodat,
                                                action      = actions[-n],
                                                action_nodeselect=actions_nodeselect[-n], 
                                                done=True,
                                                reward      = sum(GAMMA_ARR[:n] * np.array(rewards[-n:])), 
                                                next_state  = next_state,
                                                next_state_tsr=nn.functional.pad(next_state_tsr,(0,0,0,Psize)) if config['qnet']=='s2v' else zerovec,
                                                next_pyg_data  = next_pyg_data,
                                                next_state_neighbors= env.neighbors[next_state[0]] ))
            
            # update state and current solution
            current_state = next_state
            current_state_tsr = next_state_tsr
            current_pyg_data = next_pyg_data
            
            # take a gradient step
            loss = None
            if len(memory) >= config['bsize']:
                experiences = memory.sample_batch(config['bsize'])
                
                batch_states_tsrs = [e.state_tsr for e in experiences] if config['qnet']=='s2v' else [zerovec]
                batch_Ws = [ e.W for e in experiences] if config['qnet']=='s2v' else [zerovec]
                batch_pyg_data = [ e.pyg_data for e in experiences] #if 'gat' in config['qnet'] else [zerodat]
                batch_actions = [e.action_nodeselect for e in experiences]
                batch_targets = []
                
                for i, experience in enumerate(experiences):
                    target = experience.reward
                    if not experience.done:
                        with torch.no_grad():
                            _, _, best_reward = Q_func_target.get_best_action(experience.next_state_tsr, 
                                                                    experience.W,
                                                                    experience.next_pyg_data,
                                                                    experience.next_state_neighbors )
                        target += (GAMMA ** N_STEP_QL) * best_reward#.detach().cpu()#.item()
                    batch_targets.append(target)
                    
                # print('batch targets: {}'.format(batch_targets))
                loss = Q_func.batch_update(batch_states_tsrs, batch_Ws, batch_pyg_data, batch_actions, batch_targets)
                grad_update_count+=1
                losses.append(loss)
                if grad_update_count % config['tau'] == 0:
                    #Q_func_target.model.load_state_dict(torch.load(Q_func.model.state.dict()))
                    Q_func_target.model = copy.deepcopy(Q_func.model)
                    print('Target network updated, epi=',episode,'grad_update_count=',grad_update_count, 'best_medianR=',current_max_Return)



        success_ratio = len(actions_nodeselect) / max(1,env.sp.spath_length)
        path_length_ratios.append(success_ratio)
        total_rewards.append(np.sum(rewards))

        """ Save model when we reach a new low average path length
        """
        #med_length = np.median(path_length_ratios[-100:])
        if (len(total_rewards)+1) % config['bsize'] == 0: # check every batchsize episodes
            if config['optim_target']=='returns': # we seek to maximize the returns per episode
                mean_Return = int(np.mean(total_rewards[-config['bsize']:])*100)/100
                if mean_Return >= current_max_Return:
                    save_best_only = (mean_Return == current_max_Return)
                    current_max_Return = mean_Return
                    checkpoint_model(Q_net, optimizer, lr_scheduler, loss, episode, mean_Return, logdir, best_only=save_best_only)                
            elif config['optim_target']=='ratios': # we seek to minimize the path lenths per episode w.r.t. shortest path to nearest target
                mean_Ratio = int(np.mean(path_length_ratios[-10:])*100)/100
                if mean_Ratio <= current_min_Ratio:
                    save_best_only = (mean_Ratio == current_min_Ratio)
                    current_min_Ratio = mean_Ratio
                    checkpoint_model(Q_net, optimizer,d lr_scheduler, loss, episode, mean_Ratio, logdir, best_only=save_best_only)
            else:
                assert False

        writer.add_scalar("1a. epsilon", epsilon, episode)
        writer.add_scalar("1b. lr", optimizer.param_groups[0]['lr'], episode)
        writer.add_scalar("2. loss",    0 if loss is None else loss, episode)
        writer.add_scalar("3. epi_len", len(actions_nodeselect), episode)
        writer.add_scalar("4. Reward per epi", total_rewards[-1], episode)
        writer.add_scalar("5. Success ratio", success_ratio, episode)


        if episode % 10 == 0:
            print('Ep %d. Loss = %.3f / median R=%.2f/last=%.2f / median Ratio=%.2f / eps=%.4f / lr=%.4f / mem=%d / target %s walk %s.' % (
                episode, (-1 if loss is None else loss), np.mean(total_rewards[-10:]), total_rewards[-1], np.mean(path_length_ratios[-10:]), epsilon,
                Q_func.optimizer.param_groups[0]['lr'], len(memory), str(env.sp.target_nodes), str(actions_nodeselect)))
            #print(path_length_ratios[-50:])
            #found_solutions[episode] = (W.clone(), [n for n in solution])


    def _moving_avg(x, N=10):
        return np.convolve(np.array(x), np.ones((N,))/N, mode='valid')

    plt.figure(figsize=(8,5))
    plt.semilogy(_moving_avg(losses, 100))
    plt.title('Loss (moving average) during training')
    plt.ylabel('loss')
    plt.xlabel('training iteration')
    plt.savefig(logdir+'/lossplot.png')
    plt.clf()

    plt.figure(figsize=(8,5))
    plt.plot(_moving_avg(total_rewards, 10))
    plt.title('Ratio (moving average) of (estimated MVC) / (real MVC)')
    plt.ylabel('ratio')
    plt.xlabel('episode')
    plt.savefig(logdir+'/ratioplot.png')
    plt.clf()

def evaluate_tabular(logdir, config, env_all):
    num_seeds   = 1
    eps_0       = 1.
    eps_min     = 0.1
    num_iter    = 1000
    gamma       = .9
    alpha_0     = .2
    alpha_decay = 0.
    initial_Q_values = 10.


    R=[]
    for i,env in enumerate(tqdm.tqdm(env_all)):
        if len(env.world_pool)==0:
            env.world_pool=[None]
        # Learn the policy
        metrics_episode_returns = {}
        metrics_episode_lengths = {}
        metrics_avgperstep = {}
        Q_tables = {}
        policy = EpsilonGreedyPolicy(env, eps_0, eps_min, initial_Q_values)
        algos  = [q_learning_exhaustive]#,sarsa,expected_sarsa]
        for algo in algos:
            metrics_all = np.zeros((num_seeds,2,num_iter*len(env.world_pool)))
            for s in range(num_seeds):
                #seed_everthing(seed=s)
                policy.reset_epsilon()
                Q_table, metrics_singleseed, policy, _ = algo(env, policy, num_iter, discount_factor=gamma, alpha_0=alpha_0, alpha_decay=alpha_decay,print_episodes=False)
                metrics_all[s] = metrics_singleseed
                print('entries in Q table:',len(Q_table))
            
            Q_tables[algo.__name__] = Q_table
            metrics_episode_returns[algo.__name__] = metrics_all[:, 0, :]
            metrics_episode_lengths[algo.__name__] = metrics_all[:, 1, :]
            metrics_avgperstep[algo.__name__] = np.sum(
                metrics_episode_returns[algo.__name__], axis=0)/np.sum(metrics_episode_lengths[algo.__name__], axis=0)
            performance_metrics = { 'e_returns': metrics_episode_returns, 'e_lengths':metrics_episode_lengths, 'rps':metrics_avgperstep}
            
            policy.epsilon=0.
            l, returns, c = EvaluatePolicy(env, policy,env.world_pool, print_runs=False, save_plots=False, logdir=logdir, eval_arg_func=EvalArgs1, silent_mode=True)
            if i%27==0:
                if len(env.world_pool)==0:
                    plotlist=[]
                else:
                    plotlist = GetFullCoverageSample(returns, env.world_pool, bins=1, n=1)
                EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir=logdir, eval_arg_func=EvalArgs1, silent_mode=False, plot_each_timestep=False)
            R+=returns         
    
    OF = open(logdir+'/Full_result.txt', 'w')
    def printing(text):
        print(text)
        OF.write(text + "\n")
    printing('Total unique graphs evaluated: '+str(len(env_all)))
    printing('Total instances evaluated: '+str(len(R))+' Avg reward: {:.2f}'.format(np.mean(R)))
    possol=np.sum(np.array(R)>0)
    printing('Number of >0 solutions: '+str(possol)+' ({:.1f}'.format(possol/len(R)*100)+'%)')
    printing('---------------------------------------')
    for k,v in config.items():
        printing(k+' '+str(v))

def evaluate_spath_heuristic(logdir, config, env_all, n_eval=20000):
    R=[]
    S=[]
    if type(env_all) == list:
        for i,env in enumerate(tqdm.tqdm(env_all)):
            policy=ShortestPathPolicy(env,weights='equal')
            #env.encode = env.encode_nfm
            l, returns, c, solves = EvaluatePolicy(env, policy,env.world_pool, print_runs=False, save_plots=False, logdir=logdir, eval_arg_func=EvalArgs3, silent_mode=True)
            num_worlds_requested = 10
            once_every = max(1,len(env_all)//num_worlds_requested)
            if i % once_every ==0:
                plotlist = GetFullCoverageSample(returns, env.world_pool, bins=10, n=15)
                EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir=logdir, eval_arg_func=EvalArgs3, silent_mode=False, plot_each_timestep=False)
            R+=returns 
            S+=solves
            print('Env',i,'Number of instances:',len(returns),'Cumulative insances:',len(R))
    else:
    # assume env_all is a superenv
        policy=ShortestPathPolicy(env_all,weights='equal')
        full_eval_list = [i for i in range(n_eval)]
        plot_eval_list = [i for i in range(10)]
        l, returns, c, solves = EvaluatePolicy(env_all, policy, full_eval_list, print_runs=False, save_plots=False, logdir=logdir, eval_arg_func=EvalArgs3, silent_mode=True)
        EvaluatePolicy(env_all, policy, plot_eval_list, print_runs=True, save_plots=True, logdir=logdir, eval_arg_func=EvalArgs3, silent_mode=False, plot_each_timestep=False)
        R+=returns 
        S+=solves
    
    OF = open(logdir+'/Full_result.txt', 'w')
    def printing(text):
        print(text)
        OF.write(text + "\n")
    ugraphs = str(len(env_all)) if type(env_all)==list else 'N/a'
    printing('Total unique graphs evaluated: '+ugraphs)
    printing('Total instances evaluated: '+str(len(R))+' Avg reward: {:.2f}'.format(np.mean(R)))    
    num_solved=np.sum(S)
    success_rate = num_solved/len(S)
    printing('Goal reached: '+str(num_solved)+' ({:.1f}'.format(success_rate*100)+'%)')
    printing('---------------------------------------')
    for k,v in config.items():
        printing(k+' '+str(v))



def evaluate(logdir, info=False, config=None, env_all=None, eval_subdir='.', n_eval=1000):
    #Test(config)
    try:
        if env_all==None or len(env_all)==0:
            return 0,0,0,0
    except:
        pass
    Q_func, Q_net, optimizer, lr_scheduler = init_model(config,fname=logdir+'/best_model.tar')
    #Q_func.model.T = 5 ## DANGEROUS, BE CAREFUL
    logdir=logdir+'/'+eval_subdir
    #Q_func, Q_net, optimizer, lr_scheduler = init_model(config,fname='results_Phase2/CombOpt/'+affix+'/ep_350_length_7.0.tar')
    policy=GNN_s2v_Policy(Q_func)
    #policy.epsilon=0.
    #e=50 # Evaluate a specific entry
    #EvaluatePolicy(env_all[0], policy,[e], print_runs=False, save_plots=False, logdir='results_Phase2/CombOpt', eval_arg_func=EvalArgs2, silent_mode=True)
    #assert False
    R=[]
    S=[]

    if type(env_all) == list:
        for i,env in enumerate(tqdm.tqdm(env_all)):
            l, returns, c, solves = EvaluatePolicy(env, policy,env.world_pool, print_runs=False, save_plots=False, logdir=logdir, eval_arg_func=EvalArgs2, silent_mode=True)
            num_worlds_requested = 10
            once_every = max(1,len(env_all)//num_worlds_requested)
            if i % once_every ==0:
                plotlist = GetFullCoverageSample(returns, env.world_pool, bins=3, n=6)
                EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir=logdir, eval_arg_func=EvalArgs2, silent_mode=False, plot_each_timestep=False)
            R+=returns 
            S+=solves
    else:
        # assume env_all is a superenv
        full_eval_list = [i for i in range(n_eval)]
        plot_eval_list = [i for i in range(30)]
        l, returns, c, solves = EvaluatePolicy(env_all, policy, full_eval_list, print_runs=False, save_plots=False, logdir=logdir, eval_arg_func=EvalArgs2, silent_mode=True)
        EvaluatePolicy(env_all, policy, plot_eval_list, print_runs=True, save_plots=True, logdir=logdir, eval_arg_func=EvalArgs2, silent_mode=False, plot_each_timestep=False)
        R+=returns 
        S+=solves

    OF = open(logdir+'/Full_result.txt', 'w')
    def printing(text):
        print(text)
        OF.write(text + "\n")
    try:
        num_unique_graphs=len(env_all)
    except:
        num_unique_graphs=1
    num_graph_instances=len(R)
    avg_return=np.mean(R)
    num_solved=np.sum(S)
    success_rate = num_solved/len(S)
    printing('Total unique graphs evaluated: '+str(num_unique_graphs))
    printing('Total instances evaluated: '+str(num_graph_instances)+' Avg reward: {:.2f}'.format(avg_return))
    printing('Goal reached: '+str(num_solved)+' ({:.1f}'.format(success_rate*100)+'%)')
    printing('---------------------------------------')
    for k,v in config.items():
        printing(k+' '+str(v))
    return num_unique_graphs, num_graph_instances, avg_return, success_rate
