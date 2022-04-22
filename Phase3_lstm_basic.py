import copy
import gym
import modules.gnn.nfm_gen
from modules.gnn.construct_trainsets import ConstructTrainSet
from modules.ppo.models_sb3_gat2 import Gat2_ActorCriticPolicy, Gat2Extractor
from modules.ppo.ppo_wrappers import PPO_ActWrapper, PPO_ObsFlatWrapper_basic
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.sim.simdata_utils import SimulateInteractiveMode_PPO
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO#, ReccurrentPPO
import time
from torch_geometric.nn.conv import MessagePassing, GATv2Conv
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean, scatter_add
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Core framework for this approach taken from https://github.com/seungeunrho/minimalRL

def GetConfig():
    config={}
    config['train_on'] = "MemTask-U1"
    config['max_nodes'] = 8
    config['max_edges'] = 22
    config['remove_paths'] = False
    config['reject_u_duplicates'] = False
    config['solve_select'] = 'solvable'
    config['nfm_func']     = 'NFM_ev_ec_t_dt_at_um_us'
    config['edge_blocking']= True
    config['node_dim']     = modules.gnn.nfm_gen.nfm_funcs[config['nfm_func']].F
    config['demoruns']     = False
    config['qnet']         = 'gat2'
    config['norm_agg']     = True
    config['emb_dim']      = 64
    config['emb_iter_T']   = 5
    config['optim_target'] = 'returns'
    config['num_step']      = 10000
    config['obs_mask']      = "None"
    config['obs_rate']      = 1.0
    config['seed0']=0
    config['numseeds']=1
    config['rootdir']='./results/results_Phase2/Pathfinding/ppo/'+ \
                                config['train_on']+'_'+'Uon'''+'/'+ \
                                config['solve_select']+'_edgeblock'+str(config['edge_blocking'])+'/'+\
                                config['qnet']+'_normagg'+str(config['norm_agg'])
    config['logdir']        = config['rootdir'] + '/' +\
                                config['nfm_func']+'/'+ \
                                'emb'+str(config['emb_dim']) + \
                                '_itT'+str(config['emb_iter_T']) + \
                                '_nstep'+str(config['num_step'])
    config['seedrange']=range(config['seed0'], config['seed0']+config['numseeds'])    
    return config

def GetEnv():
    config = GetConfig()
    state_repr='etUte0U0'
    state_enc='nfm'
    edge_blocking = True
    env = GetCustomWorld('MemoryTaskU1Long',make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    nfm_func = modules.gnn.nfm_gen.nfm_funcs['NFM_ev_ec_t_dt_at_ustack']
    nfm_func.reinit(k=2) # set stack size
    env.redefine_nfm(nfm_func)
    env.capture_on_edges = edge_blocking
    env = PPO_ObsFlatWrapper_basic(env, obs_mask='freq', obs_rate=.2)
    env = PPO_ActWrapper(env) # apply actions as node ids
    #SimulateInteractiveMode_PPO(env, filesave_with_time_suffix=False)
    return env

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self, env):
        super(PPO, self).__init__()
        self.data = []
        self.obs_space = env.observation_space.shape[0]
        self.act_space = env.action_space.n
        self.fc1   = nn.Linear(self.obs_space, 256)
        self.fc_pi = nn.Linear(256,self.act_space)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, reachable, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        x[~reachable]=-torch.inf
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst, mask_lst = [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done, mask = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            mask_lst.append(mask)
            
        s,a,r,s_prime,done_mask,prob_a,mask = torch.stack(s_lst), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.stack(s_prime_lst), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst), \
                                          torch.tensor(mask_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, mask
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, mask = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, mask, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

class PPO_LSTM(nn.Module):
    def __init__(self, env, emb_dim=32):
        super(PPO_LSTM, self).__init__()
        self.data = []
        self.obs_space = env.observation_space.shape[0]
        self.act_space = env.action_space.n
        self.num_nodes = env.action_space.n
        self.node_dim = env.F
        self.emb_dim = emb_dim

        self.fc1   = nn.Linear(self.obs_space,emb_dim)
        self.lstm  = nn.LSTM(emb_dim,emb_dim)
        self.fc_pi = nn.Linear(emb_dim,self.act_space)
        self.fc_v  = nn.Linear(emb_dim,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden, reachable):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, self.emb_dim)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)

        x[~reachable.reshape(x.shape)]=-torch.inf
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, self.emb_dim)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst, mask_lst = [], [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done, mask = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            mask_lst.append(mask)
            
        s,a,r,s_prime,done_mask,prob_a,mask = torch.stack(s_lst), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.stack(s_prime_lst), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst), \
                                          torch.tensor(mask_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a,  h_in_lst[0], h_out_lst[0], mask
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, (h_in, c_in), (h_out, c_out), mask = self.make_batch()
        first_hidden = (h_in.detach(), c_in.detach())
        second_hidden = (h_out.detach(), c_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden, mask)
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s , td_target.detach()) #CHECK TODO

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()

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

class PPO_GNN_LSTM(nn.Module):
    def __init__(self, env, emb_dim):
        super(PPO_GNN_LSTM, self).__init__()
        self.data = []
        self.obs_space = env.observation_space.shape[0]
        self.act_space = env.action_space
        self.num_nodes = env.action_space.n
        self.node_dim = env.F
        self.emb_dim = emb_dim

        kwargs={'concat':True}
        self.gat = GATv2(
            in_channels = self.node_dim,
            hidden_channels = self.emb_dim,
            heads = 2,
            num_layers = 5,
            out_channels = self.emb_dim,
            share_weights = True,
            **kwargs
        ).to(device)      
        
        self.lstm  = nn.LSTM(self.emb_dim,self.emb_dim, device=device)
        
        # Policy network parameters
        self.theta5_pi = nn.Linear(2*self.emb_dim, 1, True, device=device)#, dtype=torch.float32)
        self.theta6_pi = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)
        self.theta7_pi = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)

        # Value network parameters        
        self.theta5_v = nn.Linear(self.emb_dim, 1, True, device=device)#, dtype=torch.float32)
        self.theta6_v = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)
        self.theta7_v = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def _deserialize(self, obs):
        # Deconstruct from an observation as defined by the PPO flattened observation wrapper
        # Args: obs [ nfm (NxF) | edge_list (Ex2) | reachable (N,) | num_nodes (1,) | max_num_nodes (1,) | num_edges (1,) | max_num_edges (1,) | node_dim (1,)]
        num_nodes, max_nodes, num_edges, max_edges, node_dim = obs[-5:].to(torch.int64).tolist()
        nfm , edge_index , reachable, _ = torch.split(obs,(node_dim * max_nodes, 2 * max_edges, max_nodes, 5), dim=0)
        nfm = nfm.reshape(max_nodes,-1)[:num_nodes]
        edge_index = edge_index.reshape(2,-1)[:,:num_edges].to(torch.int64)
        reachable = reachable.reshape(-1,1).to(torch.int64)[:num_nodes]
        return nfm, edge_index, reachable, num_nodes, max_nodes, num_edges, max_edges, node_dim


    def extract_features(self, x):
        if len(x.shape) == 1:
            x = x.view(1,-1)
        pyg_list=[]
        reachable_nodes_tensor=torch.tensor([]).to(device)
        for e in x:
            pygx, pygei, reachable_nodes, num_nodes, max_nodes, num_edges, max_edges, node_dim = self._deserialize(e)
            pyg_list.append(Data(
                pygx,
                pygei
            ))
            reachable_nodes_tensor = torch.cat((reachable_nodes_tensor, reachable_nodes.squeeze())).to(torch.bool)
        
        pyg_data = Batch.from_data_list(pyg_list)
        mu = self.gat(pyg_data.x, pyg_data.edge_index)
        return mu, pyg_data.batch, reachable_nodes_tensor

    def pi(self, x, hidden, reachable):
        x=x.to(device)
        if len(x.shape) == 1:
            x = x.view(1,-1)
        seq_len = x.shape[0]        
        mu, batch, reachable_nodes_tensor = self.extract_features(x) # mu: (seq_len * num_nodes, emb_dim)
        reachable_nodes_tensor = reachable_nodes_tensor.reshape(seq_len,self.num_nodes)
        mu = mu.reshape(seq_len,-1,self.emb_dim) # mu: (seq_len, num_nodes, emb_dim)
        mu, lstm_hidden = self.lstm(mu, hidden)  # mu: (seq_len, num_nodes, emb_dim)
        
        #mu_meanpool = scatter_mean(mu, batch, dim=1) # mu_meanpool: (seq_len, 1, emb_dim)
        mu_meanpool = mu.mean(dim=1, keepdim=True) # mu_meanpool: (seq_len, 1, emb_dim). scatter mean not needed: equal size graphs
        expander = torch.tensor([self.num_nodes]*seq_len, dtype=torch.int64, device=device)
        mu_meanpool_expanded = torch.repeat_interleave(mu_meanpool, expander, dim=0).reshape(seq_len,self.num_nodes,self.emb_dim) # (seq_len, num_nodes, emb_dim) # (seq_len, num_nodes, emb_dim)

        global_state = self.theta6_pi(mu_meanpool_expanded) # yields (seq_len, nodes in batch, emb_dim)
        local_action = self.theta7_pi(mu)  # yields (seq_len, nodes in batch, emb_dim)
        rep = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (#nodes in batch, 2*emb_dim)        
        prob_logits = self.theta5_pi(rep).squeeze(-1) # (nr_nodes in batch,)
        prob_logits[~reachable_nodes_tensor]=-torch.inf
        prob = F.softmax(prob_logits, dim=1)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x=x.to(device)
        if len(x.shape) == 1:
            x = x.view(1,-1)
        seq_len = x.shape[0]        
        mu, batch, reachable_nodes_tensor = self.extract_features(x) # mu: (seq_len * num_nodes, emb_dim)
        reachable_nodes_tensor = reachable_nodes_tensor.reshape(seq_len,self.num_nodes)
        mu = mu.reshape(seq_len,-1,self.emb_dim) # mu: (seq_len, num_nodes, emb_dim)
        mu, lstm_hidden = self.lstm(mu, hidden)
        
        #mu_meanpool = scatter_mean(mu, batch, dim=1) # mu_meanpool: (seq_len, 1, emb_dim)
        mu_meanpool = mu.mean(dim=1, keepdim=True) # mu_meanpool: (seq_len, 1, emb_dim). scatter mean not needed: equal size graphs

        #expander = torch.tensor([self.num_nodes]*seq_len, dtype=torch.int64, device=device)
        #mu_meanpool_expanded = torch.repeat_interleave(mu_meanpool, expander, dim=0).reshape(seq_len,self.num_nodes,self.emb_dim) # (seq_len, num_nodes, emb_dim)
        #global_state = self.theta6_pi(mu_meanpool_expanded) # yields (seq_len, nodes in batch, emb_dim)
        #local_action = self.theta7_pi(mu)  # yields (seq_len, nodes in batch, emb_dim)
        #v = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (#nodes in batch, 2*emb_dim)        
        v = self.theta5_v(mu_meanpool)#.squeeze()#-1) # (nr_nodes in batch,)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst, mask_lst = [], [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done, mask = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            mask_lst.append(mask)
            
        s,a,r,s_prime,done_mask,prob_a,mask = torch.stack(s_lst), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.stack(s_prime_lst), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst), \
                                          torch.tensor(mask_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a,  h_in_lst[0], h_out_lst[0], mask
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, (h_in, c_in), (h_out, c_out), mask = self.make_batch()
        first_hidden = (h_in.detach(), c_in.detach())
        second_hidden = (h_out.detach(), c_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1).cpu()
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1).cpu()
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden, mask)
            pi=pi.cpu()
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s , td_target.detach()) #CHECK TODO

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()


def Solve():
    env = GetEnv()
    model = PPO(env)
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                mask = env.action_masks()
                prob = model.pi(s, torch.tensor(mask) )
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)

                model.put_data((s, a, r/10.0, s_prime, prob[a].item(), done, mask))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()

def Solve_LSTM(env, model):
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        h_out = (
            torch.zeros([1, model.num_nodes, model.emb_dim], dtype=torch.float, device=device), 
            torch.zeros([1, model.num_nodes, model.emb_dim], dtype=torch.float, device=device)
            )
        s = env.reset()
        done = False

        while not done:
            for t in range(T_horizon):
                mask = env.action_masks()
                h_in = h_out
                prob, h_out = model.pi(s, h_in, torch.tensor(mask))
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)

                model.put_data((s, a, r/10.0, s_prime, prob[a].item(), h_in, h_out, done, mask))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()

from modules.ppo.helpfuncs import CreateEnv
from modules.rl.environments import SuperEnv
if __name__ == '__main__':
    #Solve()
    
    # basic mem task, flat observation vector
    #env = GetEnv()
    #model = PPO_LSTM(env)
    #Solve_LSTM(env,model)

    # basic mem task, nfm/ei/reachable flat vector
    env = CreateEnv('MemoryTaskU1', max_nodes=8, max_edges=22, nfm_func_name='NFM_ev_ec_t_dt_at_um_us', var_targets=None, remove_world_pool=False, apply_wrappers=True, type_obs_wrap='Flat', obs_mask='freq', obs_rate=0.2)
    env_all_list = []
    env_all_list.append(env)
    
    super_env=SuperEnv(
        [env],
        hashint2env=None,
        max_possible_num_nodes=8,
        probs=[1])

    model = PPO_GNN_LSTM(super_env, emb_dim=24)
    Solve_LSTM(env, model)