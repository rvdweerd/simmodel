import re
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us, NFM_ev_ec_t_u
from torch.distributions import Categorical
from modules.rl.rl_custom_worlds import GetCustomWorld
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# code from https://github.com/seungeunrho/minimalRL/blob/master/ppo.py
#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20
num_epi       = 10000

class PPONet(nn.Module):
    def __init__(self, config):
        super(PPONet, self).__init__()
        self.emb_dim = config['emb_dim']
        self.T = config['emb_iter_T']
        self.node_dim = config['node_dim'] #node feature dim
        self.data = []

        # Build the learnable affine maps:
        self.theta1a = nn.Linear(self.node_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta1b = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta2 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta3 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta4 = nn.Linear(1, self.emb_dim, True, dtype=torch.float32)
        self.theta5_pi = nn.Linear(2*self.emb_dim, 1, True, dtype=torch.float32)
        self.theta5_v  = nn.Linear(2*self.emb_dim, 1, True, dtype=torch.float32) # Maybe too complex, perhaps share weights with th_5_pi CHECK / TODO
        self.theta6 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta7 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
  
        #self.numTrainableParameters()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def propagate(self, xv, Ws):
        # xv: The node features (batch_size, num_nodes, node_dim)
        # Ws: The graphs (batch_size, num_nodes, num_nodes)
        
        num_nodes = xv.shape[1]
        batch_size = xv.shape[0]
        
        xv=xv.to(device)
        Ws=Ws.to(device)
        # pre-compute 1-0 connection matrices masks (batch_size, num_nodes, num_nodes)
        #conn_matrices = torch.where(Ws > 0, torch.ones_like(Ws), torch.zeros_like(Ws)).to(device)
        conn_matrices = Ws # we have only edge weights of 1

        # Graph embedding
        # Note: we first compute s1 and s3 once, as they are not dependent on mu
        mu = torch.zeros(batch_size, num_nodes, self.emb_dim, dtype=torch.float32,device=device)
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

        # we repeat the global state (summed over nodes) for each node, 
        # in order to concatenate it to local states later
        global_state = self.theta6(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1))
        local_action = self.theta7(mu)  # (batch_dim, nr_nodes, emb_dim)
        out = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (batch_dim, nr_nodes, 2*emb_dim)
        return out

    def pi(self, x, W, reachable_nodes):
        # reachable_nodes (batch_dim, nr_nodes)
        reachable_nodes=reachable_nodes.type(torch.BoolTensor)

        rep = self.propagate(x, W) # (batch_dim, nr_nodes, 2*emb_dim)
        prob_logits = self.theta5_pi(rep).squeeze(dim=2) # (batch_dim, nr_nodes)

        # mask invalid actions
        prob_logits[~reachable_nodes] = -torch.inf
        return prob_logits # returns the logits!!! (batch_dim, nr_nodes)
    
    def v(self, x, W, reachable_nodes):
        rep = self.propagate(x, W) # (batch_dim, nr_nodes, 2*emb_dim)
        rep2 = self.theta5_v(rep).squeeze(dim=2) # (batch_dim, nr_nodes)
        reachable_nodes=reachable_nodes.type(torch.BoolTensor)
        #v = rep2.sum(dim=1)
        v = rep2[reachable_nodes].sum()
        return v # (batch_dim): a value of the current graph state for each transition
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst, W_lst, reachable_nodes_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done ,W, reachable_nodes = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            W_lst.append(W)
            reachable_nodes_lst.append(reachable_nodes)
            
        s,a,r,s_prime,done_mask, prob_a = torch.stack(s_lst), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.stack(s_prime_lst), \
                                          torch.tensor(done_lst, dtype=torch.float32), torch.tensor(prob_a_lst)
        W = torch.stack(W_lst)
        reachable_nodes = torch.stack(reachable_nodes_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, W, reachable_nodes
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, W, reachable_nodes = self.make_batch()
        a=a.to(device)
        r=r.to(device)
        done_mask=done_mask.to(device)
        prob_a=prob_a.to(device)

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime, W, reachable_nodes) * done_mask
            delta = td_target - self.v(s, W, reachable_nodes)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float32).to(device)

            log_pi = self.pi(s, W, reachable_nodes)
            log_pi_a = log_pi.gather(1,a)
            ratio = torch.exp(log_pi_a - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s, W, reachable_nodes) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
def GetTrainedModel(env):
    config={}
    config['emb_dim']=64
    config['emb_iter_T']=4
    config['node_dim']=5
    model = PPONet(config).to(device)
    score = 0.0
    print_interval = 20

    for n_epi in range(num_epi):
        env.reset()
        s = env.nfm
        done = False
        while not done:
            prob_logits = model.pi(s.unsqueeze(0), env.sp.W.unsqueeze(0), env.sp.W[env.state[0]].unsqueeze(0))
            m = Categorical(logits = prob_logits.squeeze())
            a = m.sample().item()
            _, r, done, info = env.step(a)
            s_prime = env.nfm

            model.put_data(( s, a, r/10.0, s_prime, m.probs[a].item(), done, env.sp.W, env.sp.W[env.state[0]]))
            s = s_prime

            score += r
            if done:
                break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    world_name='MetroU3_e17tborder_FixedEscapeInit'
    scenario_name='TrainMetro'
    state_repr='etUte0U0'
    state_enc='nfm'
    nfm_funcs = {
    'NFM_ev_ec_t'       : NFM_ev_ec_t(),
    'NFM_ec_t'          : NFM_ec_t(),
    'NFM_ev_t'          : NFM_ev_t(),
    'NFM_ev_ec_t_u'     : NFM_ev_ec_t_u(),
    'NFM_ev_ec_t_um_us' : NFM_ev_ec_t_um_us(),
    }
    nfm_func=nfm_funcs['NFM_ev_ec_t_um_us']
    edge_blocking = True

    env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    env.redefine_nfm(nfm_func)
    env.capture_on_edges = edge_blocking

    model = GetTrainedModel(env)