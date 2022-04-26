import copy
from pathlib import Path
import numpy as np
import modules.gnn.nfm_gen
from modules.ppo.ppo_wrappers import PPO_ActWrapper, PPO_ObsFlatWrapper_basic
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.sim.simdata_utils import SimulateInteractiveMode_PPO
from stable_baselines3 import PPO
import time
from torch_geometric.nn.conv import MessagePassing, GATv2Conv
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.data import Data, Batch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Core framework for this approach taken from https://github.com/seungeunrho/minimalRL

def GetLocalConfig():
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
    config['num_step']      = 100000
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
    config = GetLocalConfig()
    state_repr='etUte0U0'
    state_enc='nfm'
    edge_blocking = True
    #env = GetCustomWorld('MemoryTaskU1Long',make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    env = GetCustomWorld('Manhattan3x3_WalkAround',make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    nfm_func = modules.gnn.nfm_gen.nfm_funcs['NFM_ev_ec_t_dt_at_ustack']
    nfm_func.reinit(k=2) # set stack size
    env.redefine_nfm(nfm_func)
    env.capture_on_edges = edge_blocking
    env = PPO_ObsFlatWrapper_basic(env, obs_mask='freq', obs_rate=.2)
    env = PPO_ActWrapper(env) # apply actions as node ids
    #SimulateInteractiveMode_PPO(env, filesave_with_time_suffix=False)
    return env

class PPO(nn.Module):
    def __init__(self, env):
        super(PPO, self).__init__()
        self.data = []
        self.obs_space = env.observation_space.shape[0]
        self.act_space = env.action_space.n
        self.fc1   = nn.Linear(self.obs_space, 256)
        self.fc_pi = nn.Linear(256,self.act_space)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)
        self.K_epoch  = 2
        self.gamma    = .98
        self.lmbda    = .95
        self.eps_clip = 0.1

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

        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, mask, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
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
        self.gamma    = .98
        self.eps_clip = 0.1        

        self.fc1   = nn.Linear(self.obs_space,emb_dim)
        self.lstm  = nn.LSTM(emb_dim,emb_dim)
        self.fc_pi = nn.Linear(emb_dim,self.act_space)
        self.fc_v  = nn.Linear(emb_dim,1)
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)

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

        for i in range(self.num_epochs):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + self.gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden, mask)
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
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

class PPO_GNN_Model(nn.Module):
    # Base class for PPO-GNN-LSTM implementations
    def __init__(self, config, hp, tp):
        super(PPO_GNN_Model, self).__init__()
        self.hp = hp
        self.tp = tp
        self.config = config
        self.data = []
        self.ei = []
        #self.obs_space = env.observation_space.shape[0]
        #self.act_space = env.action_space
        #self.num_nodes = env.action_space.n
        self.node_dim = hp.node_dim#env.F
        self.emb_dim = hp.emb_dim
        self.num_rollouts = hp.parallel_rollouts
        self.num_epochs = hp.batch_size
        assert config['lstm_type'] in ['None','EMB','FE','Dual','DualCC']
        assert config['qnet'] == 'gat2'
        assert config['critic'] in ['q','v']
        Path(self.tp["base_checkpoint_path"]).mkdir(parents=True, exist_ok=True)

    def _deserialize(self, obs):
        # Deconstruct from an observation as defined by the PPO flattened observation wrapper
        # Args: obs [ nfm (NxF) | edge_list (Ex2) | reachable (N,) | num_nodes (1,) | max_num_nodes (1,) | num_edges (1,) | max_num_edges (1,) | node_dim (1,)]
        num_nodes, max_nodes, num_edges, max_edges, node_dim = obs[-5:].to(torch.int64).tolist()
        nfm , edge_index , reachable, _ = torch.split(obs,(node_dim * max_nodes, 2 * max_edges, max_nodes, 5), dim=0)
        nfm = nfm.reshape(max_nodes,-1)[:num_nodes]
        edge_index = edge_index.reshape(2,-1)[:,:num_edges].to(torch.int64)
        reachable = reachable.reshape(-1,1).to(torch.int64)[:num_nodes]
        return nfm, edge_index, reachable, num_nodes, max_nodes, num_edges, max_edges, node_dim

    def put_data(self, transition_buffer, ei=None):
        self.data.append(transition_buffer)
        self.ei.append(ei)

    def checkpoint(self, n_epi, mean_Return, mode=None):
        if mode == 'best':
            print('...New best det results, saving model, it=',n_epi,'avg_return=',mean_Return)
            fname = self.tp["base_checkpoint_path"] + "best_model.tar"
            announce = 'BEST_' if mode=='best' else '...._'
            OF = open(self.tp["base_checkpoint_path"]+'/model_best_save_history.txt', 'a')
            OF.write(announce+'timestep:'+str(n_epi)+', avg det res:'+str(mean_Return)+'\n')
            OF.close()            
        elif mode == 'last':
            fname = self.tp["base_checkpoint_path"] + "model_"+str(n_epi)+".tar"
        else: return
        #checkpoint = torch.load(fname)
        #self.load_state_dict(checkpoint['weights'])
        torch.save({
            'weights':self.state_dict(),
            'optimizer':self.optimizer.state_dict(),
        }, fname)

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

class PPO_GNN_Single_LSTM(PPO_GNN_Model):
    # Single LSTM acting either on node features (lstm_type='FE') or on embedding ('EMB'). LSTM can be disabled ('None')
    def __init__(self, config, hp, tp):
        super(PPO_GNN_Single_LSTM, self).__init__(config, hp, tp)
        self.description="PPO policy, GATv2 extractor, lstm, action masking"
        assert config['lstm_type'] in ['None','EMB','FE']

        kwargs={'concat':config['gat_concat']}
        self.gat = GATv2(
            in_channels = self.node_dim,
            hidden_channels = self.emb_dim,
            heads = config['gat_heads'],
            num_layers = config['emb_iterT'],
            out_channels = self.emb_dim,
            share_weights = config['gat_share_weights'],
            **kwargs
        ).to(device)      

        if config['lstm_type'] == 'EMB':
            self.lstm_on=True
            self.lstm  = nn.LSTM(self.emb_dim,self.emb_dim, device=device)
        elif config['lstm_type'] == 'FE':
            self.lstm_on=True
            self.lstm  = nn.LSTM(self.node_dim,self.node_dim, device=device)
        elif config['lstm_type'] == 'None':
            self.lstm_on = False
        
        # Policy network parameters
        self.theta5_pi = nn.Linear(2*self.emb_dim, 1, True, device=device)#, dtype=torch.float32)
        self.theta6_pi = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)
        self.theta7_pi = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)

        if config['critic'] ==  'v':
            # Value network parameters OPTION 2: LINEAR OPERATOR       
            self.theta5_v = nn.Linear(self.emb_dim, 1, True, device=device)#, dtype=torch.float32)
        elif config['critic'] ==  'q':
            # Value network parameters OPTION 2: MAX OPERATOR       
            self.theta5_v = nn.Linear(2*self.emb_dim, 1, True, device=device)#, dtype=torch.float32)
            self.theta6_v = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)
            self.theta7_v = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)

        self.optimizer = optim.Adam(self.parameters(), lr=self.hp.learning_rate)

    def process_input(self, nfm, ei, reachable, hidden):
        nfm = nfm.to(device)
        reachable = reachable.to(device)
        ei = ei.to(device)
        if len(nfm.shape) == 2:
            nfm = nfm[None,:,:]
            reachable = reachable[None,:]
        seq_len = nfm.shape[0]
        num_nodes = nfm.shape[1]
        if self.config['lstm_type'] == 'FE':
            nfm, lstm_hidden = self.lstm(nfm, hidden)
        else:
            lstm_hidden = hidden
        return num_nodes, seq_len, nfm, ei, reachable, lstm_hidden

    def create_embeddings(self, seq_len, nfm, ei, lstm_hidden):
        pyg_list=[]
        for e in nfm:
            pyg_list.append(Data(e, ei))
        pyg_data = Batch.from_data_list(pyg_list)
        mu = self.gat(pyg_data.x, pyg_data.edge_index)
        
        mu = mu.reshape(seq_len, -1, self.emb_dim) # mu: (seq_len, num_nodes, emb_dim)        
        if self.config['lstm_type'] == 'EMB':
            mu, lstm_hidden = self.lstm(mu, lstm_hidden)  # mu: (seq_len, num_nodes, emb_dim)
        
        return mu, lstm_hidden

    def pi(self, nfm, ei, reachable, hidden):
        num_nodes, seq_len, nfm, ei, reachable, lstm_hidden = self.process_input(nfm, ei, reachable, hidden)
        mu, lstm_hidden = self.create_embeddings(seq_len, nfm, ei, lstm_hidden)
        
        mu_meanpool = mu.mean(dim=1, keepdim=True) # mu_meanpool: (seq_len, 1, emb_dim). scatter mean not needed: equal size graphs
        
        expander = torch.tensor([num_nodes]*seq_len, dtype=torch.int64, device=device)
        mu_meanpool_expanded = torch.repeat_interleave(mu_meanpool, expander, dim=0).reshape(seq_len, num_nodes, self.emb_dim) # (seq_len, num_nodes, emb_dim) # (seq_len, num_nodes, emb_dim)
        global_state = self.theta6_pi(mu_meanpool_expanded) # yields (seq_len, nodes in batch, emb_dim)
        local_action = self.theta7_pi(mu)  # yields (seq_len, nodes in batch, emb_dim)
        rep = F.relu(torch.cat((global_state, local_action), dim=2)) # concat creates (#nodes in batch, 2*emb_dim)        
        
        prob_logits = self.theta5_pi(rep).squeeze(-1) # (nr_nodes in batch,)
        prob_logits[~reachable] = -torch.inf
        prob = F.softmax(prob_logits, dim=1)

        return prob, lstm_hidden
    
    def v(self, nfm, ei, reachable, hidden):
        num_nodes, seq_len, nfm, ei, reachable, lstm_hidden = self.process_input(nfm, ei, reachable, hidden)
        mu, lstm_hidden = self.create_embeddings(seq_len, nfm, ei, lstm_hidden)
        
        mu_meanpool = mu.mean(dim=1, keepdim=True) # mu_meanpool: (seq_len, 1, emb_dim). scatter mean not needed: equal size graphs

        # OPTION 1: linear operator
        if self.config['critic'] == 'v':
            v = self.theta5_v(mu_meanpool).squeeze(-1) # (seq_len, 1)

        # # OPTION 2: max operator
        if self.config['critic'] == 'q':
            expander = torch.tensor([num_nodes]*seq_len, dtype=torch.int64, device=device)
            mu_meanpool_expanded = torch.repeat_interleave(mu_meanpool, expander, dim=0).reshape(seq_len,num_nodes,self.emb_dim) # (seq_len, num_nodes, emb_dim)
            global_state = self.theta6_v(mu_meanpool_expanded) # yields (seq_len, nodes in batch, emb_dim)
            local_action = self.theta7_v(mu)  # yields (seq_len, nodes in batch, emb_dim)
            rep = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (#nodes in batch, 2*emb_dim)
            
            qvals = self.theta5_v(rep).squeeze(-1)  #(seq_len, nr_nodes in batch)
            qvals[~reachable] = -torch.inf
            v = qvals.max(dim=1)[0].unsqueeze(-1)
        
        return v, lstm_hidden
         
    def make_batch(self):
        batch = []
        for i in range(self.num_rollouts):
            s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst, reachable_lst, reachable_prime_lst = [], [], [], [], [], [], [], [], [], []
            for transition in self.data[i]:
                s, a, r, s_prime, prob_a, h_in, h_out, done, reachable, reachable_prime = transition
                
                s_lst.append(s)
                a_lst.append([a])
                r_lst.append([r])
                s_prime_lst.append(s_prime)
                prob_a_lst.append([prob_a])
                h_in_lst.append(h_in)
                h_out_lst.append(h_out)
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                reachable_lst.append(reachable)
                reachable_prime_lst.append(reachable_prime)
                
            s,a,r,s_prime,done_mask,prob_a,reachable,reachable_prime = torch.stack(s_lst), torch.tensor(a_lst), \
                                            torch.tensor(r_lst), torch.stack(s_prime_lst), \
                                            torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst), \
                                            torch.stack(reachable_lst), torch.stack(reachable_prime_lst)
            batch.append((s, a, r, s_prime, done_mask, prob_a,  h_in_lst[0], h_out_lst[0], reachable, reachable_prime, self.ei[i]))
        self.data = []
        self.ei=[]
        return batch#s, a, r, s_prime, done_mask, prob_a,  h_in_lst[0], h_out_lst[0], mask
        
    def train_net(self):
        batch = self.make_batch()
        for _ in range(self.num_epochs):
            loss_tsr = torch.tensor([])#.to(device)
            for j in range(self.num_rollouts):
                nfm, a, r, nfm_prime, done_mask, prob_a, (h_in, c_in), (h_out, c_out), reachable, reachable_prime, ei = batch[j]
                first_hidden = (h_in.detach(), c_in.detach())
                second_hidden = (h_out.detach(), c_out.detach())

                v_prime = self.v(nfm_prime, ei, reachable_prime, second_hidden)[0].cpu()#.squeeze(1).cpu()
                #v_prime = self.v(s_prime, second_hidden)[0].unsqueeze(-1).cpu()
                td_target = r + self.hp.discount * v_prime * done_mask
                v_s = self.v(nfm, ei, reachable, first_hidden)[0].cpu()#.squeeze(1).cpu()
                #v_s = self.v(s, first_hidden)[0].unsqueeze(-1).cpu()
                delta = td_target - v_s
                delta = delta.detach().numpy()

                advantage_lst = []
                advantage = 0.0
                for item in delta[::-1]:
                    advantage = self.hp.discount * self.hp.gae_lambda * advantage + item[0]
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = torch.tensor(advantage_lst, dtype=torch.float)

                pi, _ = self.pi(nfm, ei, reachable, first_hidden)
                pi=pi.cpu()
                pi_a = pi.squeeze(1).gather(1,a)
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.hp.ppo_clip, 1 + self.hp.ppo_clip) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s , td_target.detach()) #CHECK TODO
                loss_tsr = torch.concat((loss_tsr,loss.squeeze(-1)))
            self.optimizer.zero_grad()
            loss_tsr.mean().backward(retain_graph=True)
            self.optimizer.step()

    def learn(self, env, it0=0, best_result=-1e6):
        score = 0.0
        counter = 0
        validation_interval = 50
        current_max_Return  = best_result

        for n_epi in range(it0, self.config['num_step'] ):
            R=0
            h_out = (torch.zeros(1), torch.zeros(1)) # dummy hidden state if lstm is disabled
            gathertimes=[]
            traintimes=[]
            start_gather_time = time.time()
            for t in range(self.num_rollouts):
                s = env.reset()
                
                if self.lstm_on:
                    h_out = (
                        torch.zeros([1, env.sp.V, self.lstm.hidden_size], dtype=torch.float, device=device), 
                        torch.zeros([1, env.sp.V, self.lstm.hidden_size], dtype=torch.float, device=device)
                        )
                done = False
                transitions=[]
                
                while not done:
                    #mask = env.action_masks()
                    h_in = h_out
                    prob, h_out = self.pi(s['nfm'], s['ei'], s['reachable'], h_in)
                    prob = prob.view(-1)
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, done, info = env.step(a)

                    transitions.append((s['nfm'], a, r/10.0, s_prime['nfm'], prob[a].item(), h_in, h_out, done, s['reachable'], s_prime['reachable']))
                    s = s_prime

                    score += r
                    R += r
                    if done:
                        break
                self.put_data(transitions, s['ei'])
                transitions = []
            counter+=1
            end_gather_time = time.time()
            start_train_time = time.time()
            self.train_net()
            end_train_time = time.time() 
            gathertimes.append(end_gather_time-start_gather_time)
            traintimes.append(end_train_time-start_train_time)

            self.tp['writer'].add_scalar('return_per_epi', R/self.num_rollouts, n_epi)
            
            if n_epi % self.tp['checkpoint_frequency']==0 and n_epi != it0:
                self.checkpoint(n_epi,score/counter/self.num_rollouts,mode='last')
            if n_epi % validation_interval == 0 and n_epi != it0:
                mean_Return = score/validation_interval/self.num_rollouts
                if mean_Return >= current_max_Return:            
                    current_max_Return = mean_Return
                    self.checkpoint(n_epi,mean_Return,mode='best')
                print("# of episode :{}, avg score : {:.1f}, gather time per iter: {:.1f}, train time per iter: {:.1f}".format(n_epi, mean_Return, np.mean(gathertimes), np.mean(traintimes)))
                counter=0
                score = 0.0
        env.close()
        return mean_Return

class PPO_GNN_Dual_LSTM(PPO_GNN_Model):
    def __init__(self, env, config, hp, tp):
        super(PPO_GNN_Dual_LSTM, self).__init__(env, config, hp, tp)
        self.description="PPO policy, GATv2 extractor, Dual lstm, action masking"
        assert config['lstm_type'] in ['Dual','DualCC']
        kwargs={'concat':config['gat_concat']}
        self.gat = GATv2(
            in_channels = self.node_dim,
            hidden_channels = self.emb_dim,
            heads = config['gat_heads'],
            num_layers = config['emb_iterT'],
            out_channels = self.emb_dim,
            share_weights = config['gat_share_weights'],
            **kwargs
        ).to(device)      

        if config['lstm_type'] == 'Dual':
            self.lstm_pi  = nn.LSTM(self.emb_dim, self.emb_dim, device=device)
            self.lstm_v   = nn.LSTM(self.emb_dim, self.emb_dim, device=device)
        if config['lstm_type'] == 'DualCC':
            self.lstm_pi  = nn.LSTM(2*self.emb_dim, 2*self.emb_dim, device=device)
            if config['critic'] ==  'v':
                self.lstm_v   = nn.LSTM(self.emb_dim, self.emb_dim, device=device)        
            else:
                self.lstm_v   = nn.LSTM(2*self.emb_dim, 2*self.emb_dim, device=device)        
        
        # Policy network parameters
        self.theta5_pi = nn.Linear(2*self.emb_dim, 1, True, device=device)#, dtype=torch.float32)
        self.theta6_pi = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)
        self.theta7_pi = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)

        if config['critic'] ==  'v':
            # Value network parameters OPTION 2: LINEAR OPERATOR       
            self.theta5_v = nn.Linear(self.emb_dim, 1, True, device=device)#, dtype=torch.float32)
        elif config['critic'] ==  'q':
            # Value network parameters OPTION 2: MAX OPERATOR       
            self.theta5_v = nn.Linear(2*self.emb_dim, 1, True, device=device)#, dtype=torch.float32)
            self.theta6_v = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)
            self.theta7_v = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)

        self.optimizer = optim.Adam(self.parameters(), lr=self.hp.learning_rate)

    def process_input(self, nfm, ei, reachable):
        nfm = nfm.to(device)
        reachable = reachable.to(device)
        ei = ei.to(device)
        if len(nfm.shape) == 2:
            nfm = nfm[None,:,:]
            reachable = reachable[None,:]
        seq_len = nfm.shape[0]
        num_nodes = nfm.shape[1]
        return num_nodes, seq_len, nfm, ei, reachable

    def create_embeddings(self, seq_len, nfm, ei):
        pyg_list=[]
        for e in nfm:
            pyg_list.append(Data(e, ei))
        pyg_data = Batch.from_data_list(pyg_list)
        mu = self.gat(pyg_data.x, pyg_data.edge_index)
        
        mu = mu.reshape(seq_len, -1, self.emb_dim) # mu: (seq_len, num_nodes, emb_dim)        
        
        return mu

    def pi(self, nfm, ei, reachable, hidden):
        num_nodes, seq_len, nfm, ei, reachable = self.process_input(nfm, ei, reachable)
        mu = self.create_embeddings(seq_len, nfm, ei)
        if self.config['lstm_type'] == 'Dual':
            mu, lstm_hidden = self.lstm_pi(mu, hidden)  # mu: (seq_len, num_nodes, emb_dim)

        mu_meanpool = mu.mean(dim=1, keepdim=True) # mu_meanpool: (seq_len, 1, emb_dim). scatter mean not needed: equal size graphs       
        expander = torch.tensor([num_nodes]*seq_len, dtype=torch.int64, device=device)
        mu_meanpool_expanded = torch.repeat_interleave(mu_meanpool, expander, dim=0).reshape(seq_len, num_nodes, self.emb_dim) # (seq_len, num_nodes, emb_dim) # (seq_len, num_nodes, emb_dim)
        global_state = self.theta6_pi(mu_meanpool_expanded) # yields (seq_len, nodes in batch, emb_dim)
        local_action = self.theta7_pi(mu)  # yields (seq_len, nodes in batch, emb_dim)
        rep = F.relu(torch.cat((global_state, local_action), dim=2)) # concat creates (#nodes in batch, 2*emb_dim)        
        if self.config['lstm_type'] == 'DualCC':
            rep, lstm_hidden = self.lstm_pi(rep, hidden)  

        prob_logits = self.theta5_pi(rep).squeeze(-1) # (nr_nodes in batch,)
        prob_logits[~reachable] = -torch.inf
        prob = F.softmax(prob_logits, dim=1)

        return prob, lstm_hidden
    
    def v(self, nfm, ei, reachable, hidden):
        num_nodes, seq_len, nfm, ei, reachable = self.process_input(nfm, ei, reachable)
        mu = self.create_embeddings(seq_len, nfm, ei)
        if self.config['lstm_type'] == 'Dual':        
            mu, lstm_hidden = self.lstm_v(mu, hidden)  # mu: (seq_len, num_nodes, emb_dim)

        mu_meanpool = mu.mean(dim=1, keepdim=True) # mu_meanpool: (seq_len, 1, emb_dim). scatter mean not needed: equal size graphs

        # OPTION 1: linear operator
        if self.config['critic'] == 'v':
            if self.config['lstm_type'] == 'DualCC':
                mu_meanpool, lstm_hidden = self.lstm_v(mu_meanpool, hidden)
            v = self.theta5_v(mu_meanpool).squeeze(-1) # (seq_len, 1)

        # # OPTION 2: max operator
        if self.config['critic'] == 'q':
            expander = torch.tensor([num_nodes]*seq_len, dtype=torch.int64, device=device)
            mu_meanpool_expanded = torch.repeat_interleave(mu_meanpool, expander, dim=0).reshape(seq_len,num_nodes,self.emb_dim) # (seq_len, num_nodes, emb_dim)
            global_state = self.theta6_v(mu_meanpool_expanded) # yields (seq_len, nodes in batch, emb_dim)
            local_action = self.theta7_v(mu)  # yields (seq_len, nodes in batch, emb_dim)
            rep = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (#nodes in batch, 2*emb_dim)
            if self.config['lstm_type'] == 'DualCC':
                rep, lstm_hidden = self.lstm_v(rep, hidden)

            qvals = self.theta5_v(rep).squeeze(-1)  #(seq_len, nr_nodes in batch)
            qvals[~reachable] = -torch.inf
            v = qvals.max(dim=1)[0].unsqueeze(-1)
        
        return v, lstm_hidden
         
    def make_batch(self):
        batch = []
        for i in range(self.num_rollouts):
            s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_pi_in_lst, h_pi_out_lst, h_v_in_lst, h_v_out_lst, done_lst, reachable_lst, reachable_prime_lst = [], [], [], [], [], [], [], [], [], [], [], []
            for transition in self.data[i]:
                s, a, r, s_prime, prob_a, h_pi_in, h_pi_out, h_v_in, h_v_out, done, reachable, reachable_prime = transition
                
                s_lst.append(s)
                a_lst.append([a])
                r_lst.append([r])
                s_prime_lst.append(s_prime)
                prob_a_lst.append([prob_a])
                h_pi_in_lst.append(h_pi_in)
                h_pi_out_lst.append(h_pi_out)
                h_v_in_lst.append(h_v_in)
                h_v_out_lst.append(h_v_out)
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                reachable_lst.append(reachable)
                reachable_prime_lst.append(reachable_prime)
                
            s,a,r,s_prime,done_mask,prob_a,reachable,reachable_prime = torch.stack(s_lst), torch.tensor(a_lst), \
                                            torch.tensor(r_lst), torch.stack(s_prime_lst), \
                                            torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst), \
                                            torch.stack(reachable_lst), torch.stack(reachable_prime_lst)
            batch.append((s, a, r, s_prime, done_mask, prob_a, h_pi_in_lst[0], h_pi_out_lst[0], h_v_in_lst[0], h_v_out_lst[0], reachable, reachable_prime, self.ei[i]))
        self.data = []
        self.ei=[]
        return batch#s, a, r, s_prime, done_mask, prob_a,  h_in_lst[0], h_out_lst[0], mask
        
    def train_net(self):
        batch = self.make_batch()
        for _ in range(self.num_epochs):
            loss_tsr = torch.tensor([])#.to(device)
            for j in range(self.num_rollouts):
                nfm, a, r, nfm_prime, done_mask, prob_a, (h_pi_in, c_pi_in), (h_pi_out, c_pi_out), (h_v_in, c_v_in), (h_v_out, c_v_out), reachable, reachable_prime, ei = batch[j]
                first_hidden_pi = (h_pi_in.detach(), c_pi_in.detach())
                second_hidden_pi = (h_pi_out.detach(), c_pi_out.detach())
                first_hidden_v = (h_v_in.detach(), c_v_in.detach())
                second_hidden_v = (h_v_out.detach(), c_v_out.detach()) 

                v_prime = self.v(nfm_prime, ei, reachable_prime, second_hidden_v)[0].cpu()#.squeeze(1).cpu()
                #v_prime = self.v(s_prime, second_hidden)[0].unsqueeze(-1).cpu()
                td_target = r + self.hp.discount * v_prime * done_mask
                v_s = self.v(nfm, ei, reachable, first_hidden_v)[0].cpu()#.squeeze(1).cpu()
                #v_s = self.v(s, first_hidden)[0].unsqueeze(-1).cpu()
                delta = td_target - v_s
                delta = delta.detach().numpy()

                advantage_lst = []
                advantage = 0.0
                for item in delta[::-1]:
                    advantage = self.hp.discount * self.hp.gae_lambda * advantage + item[0]
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = torch.tensor(advantage_lst, dtype=torch.float)

                pi, _ = self.pi(nfm, ei, reachable, first_hidden_pi)
                pi=pi.cpu()
                pi_a = pi.squeeze(1).gather(1,a)
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.hp.ppo_clip, 1 + self.hp.ppo_clip) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s , td_target.detach()) #CHECK TODO
                loss_tsr = torch.concat((loss_tsr,loss.squeeze(-1)))
            self.optimizer.zero_grad()
            loss_tsr.mean().backward(retain_graph=True)
            self.optimizer.step()

    def learn(self, env, it0=0, best_result=-1e6):
        score = 0.0
        counter=0
        validation_interval = 50
        current_max_Return  = best_result

        for n_epi in range(it0, self.config['num_step'] ):
            R=0
            h_out = (torch.zeros(1), torch.zeros(1)) # dummy hidden state if lstm is disabled
            gathertimes=[]
            traintimes=[]
            start_gather_time = time.time()
            for t in range(self.num_rollouts):
                s = env.reset()
                lstm_v_batchdim = 1 if (self.config['lstm_type']=='DualCC' and self.config['critic']=='v') else env.sp.V
                h_out_pi = (
                    torch.zeros([1, env.sp.V, self.lstm_pi.hidden_size], dtype=torch.float, device=device), 
                    torch.zeros([1, env.sp.V, self.lstm_pi.hidden_size], dtype=torch.float, device=device)
                    )
                h_out_v = (
                        torch.zeros([1, lstm_v_batchdim, self.lstm_v.hidden_size], dtype=torch.float, device=device), 
                        torch.zeros([1, lstm_v_batchdim, self.lstm_v.hidden_size], dtype=torch.float, device=device)
                        )                    
                 
                done = False
                transitions=[]
                
                while not done:
                    #mask = env.action_masks()
                    h_in_pi = h_out_pi
                    h_in_v = h_out_v
                    with torch.no_grad():
                        prob, h_out_pi = self.pi(s['nfm'], s['ei'], s['reachable'], h_in_pi)
                        v, h_out_v = self.v(s['nfm'], s['ei'], s['reachable'], h_in_v)
                    prob = prob.view(-1)
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, done, info = env.step(a)

                    transitions.append((s['nfm'], a, r/10.0, s_prime['nfm'], prob[a].item(), h_in_pi, h_out_pi, h_in_v, h_out_v, done, s['reachable'], s_prime['reachable']))
                    s = s_prime

                    score += r
                    R += r
                    if done:
                        break
                self.put_data(transitions, s['ei'])
                transitions = []
            counter+=1
            end_gather_time = time.time()
            start_train_time = time.time()
            self.train_net()
            end_train_time = time.time() 
            gathertimes.append(end_gather_time-start_gather_time)
            traintimes.append(end_train_time-start_train_time)

            self.tp['writer'].add_scalar('return_per_epi', R/self.num_rollouts, n_epi)
            
            if n_epi % self.tp['checkpoint_frequency']==0 and n_epi != it0:
                self.checkpoint(n_epi,score/counter/self.num_rollouts,mode='last')
            if n_epi % validation_interval == 0 and n_epi != it0:
                mean_Return = score/validation_interval/self.num_rollouts
                if mean_Return >= current_max_Return:            
                    current_max_Return = mean_Return
                    self.checkpoint(n_epi,mean_Return,mode='best')
                print("# of episode :{}, avg score : {:.1f}, gather time per iter: {:.1f}, train time per iter: {:.1f}".format(n_epi, mean_Return, np.mean(gathertimes), np.mean(traintimes)))
                score = 0.0
                counter=0
        env.close()
        return mean_Return




def Solve(env, model):
    #env = GetEnv()
    #model = PPO(env)
    score = 0.0
    validation_interval = 20
    T_horizon     = 200

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

        if n_epi % validation_interval == 0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/validation_interval))
            score = 0.0
    env.close()

def Solve_LSTM(env, model):
    score = 0.0
    validation_interval = 20
    T_horizon     = 200

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

        if n_epi % validation_interval == 0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/validation_interval))
            score = 0.0
    env.close()

