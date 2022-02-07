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

class PPO(nn.Module):
    def __init__(self, dim_obs=4, dim_actions=2):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(dim_obs,256)
        self.fc_pi = nn.Linear(256,dim_actions)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, W = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime, W) * done_mask
            delta = td_target - self.v(s, W)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, W, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
def GetTrainedModel(env):
    model = PPO(env.state_encoding_dim, env.action_space.n)
    score = 0.0
    print_interval = 20

    for n_epi in range(num_epi):
        env.reset()
        s = env.nfm
        done = False
        while not done:
            prob = model.pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()
            _, r, done, info = env.step(a)
            s_prime = env.nfm

            model.put_data(( s, a, r/10.0, s_prime, prob[a].item(), done, env.sp.W))
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