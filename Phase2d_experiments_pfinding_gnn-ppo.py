import re
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us, NFM_ev_ec_t_u
from torch.distributions import Categorical
from modules.dqn.dqn_utils import seed_everything
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.sim.simdata_utils import SimulateInteractiveMode
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#import matplotlib.pyplot as plt


# code from https://github.com/seungeunrho/minimalRL/blob/master/ppo.py


class PPONet(nn.Module):
    def __init__(self, config):
        super(PPONet, self).__init__()
        self.emb_dim        = config['emb_dim']
        self.T              = config['emb_iter_T']
        self.node_dim       = config['node_dim'] #node feature dim
        self.learning_rate  = config['learning_rate'] 
        self.gamma          = config['gamma']         
        self.lmbda          = config['lmbda']         
        self.eps_clip       = config['eps_clip']   
        self.K_epoch        = config['K_epoch']     
        self.num_epi        = config['num_epi']     
        self.count=0
        self.data = []

        # Build the learnable affine maps:
        self.theta1a = nn.Linear(self.node_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta1b = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta2 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta3 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta4 = nn.Linear(1, self.emb_dim, True, dtype=torch.float32)
        self.theta5_pi = nn.Linear(2*self.emb_dim, 1, True, dtype=torch.float32)
        #self.theta5_pi2 = nn.Linear(self.emb_dim, 1, True, dtype=torch.float32) # Maybe too complex, perhaps share weights with th_5_pi CHECK / TODO 
        self.theta5_v1 = nn.Linear(2*self.emb_dim, 1, True, dtype=torch.float32)
        #self.theta5_v2  = nn.Linear(self.emb_dim, 1, True, dtype=torch.float32) # Maybe too complex, perhaps share weights with th_5_pi CHECK / TODO
        self.theta6 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta7 = nn.Linear(self.emb_dim, self.emb_dim, True, dtype=torch.float32)
  
        #self.thetav1 = nn.Linear(self.emb_dim*2,self.emb_dim, True, dtype=torch.float32)
        #self.thetav2 = nn.Linear(self.emb_dim, 1, True, dtype=torch.float32)
        #self.numTrainableParameters()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

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
        #rep = F.relu(self.theta5_pi1(rep))
        prob_logits = self.theta5_pi(rep).squeeze(dim=2) # (batch_dim, nr_nodes)

        # mask invalid actions
        prob_logits[~reachable_nodes] = -torch.inf
        #print(self.count,prob_logits)
        self.count+=1
        return prob_logits # returns the logits!!! (batch_dim, nr_nodes)

    
    def v(self, x, W, reachable_nodes):
        rep = self.propagate(x, W) # (batch_dim, nr_nodes, 2*emb_dim)
        
        # # OPTION 1: aggregate (mean = best) and project
        # #rep2 = torch.max(rep,dim=1)[0]
        # rep = rep.mean(dim=1)
        # v = self.theta5_v1(rep).squeeze()

        # OPTION 2: mimic Q-function and take the max in the reachable node dimension
        #rep = F.relu(self.theta5_v1(rep))
        rep = self.theta5_v1(rep).squeeze(dim=2) # (batch_dim, nr_nodes)
        # optional: ONLY LOOK AT DIRECT NEIGHBORHOOD
        #reachable_nodes=reachable_nodes.type(torch.BoolTensor).to(device)
        #rep[~reachable_nodes] = -torch.inf
        #v=torch.max(rep,dim=1)[0]
        #v = (rep * reachable_nodes).mean(dim=1)
        v = rep.mean(dim=1)
        return v # (batch_dim): a value of the current graph state for each transition
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst, W_lst, reachable_nodes_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done ,W, reachable_nodes = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append(done_mask)
            W_lst.append(W)
            reachable_nodes_lst.append(reachable_nodes)
            
        s,a,r,s_prime,done_mask, prob_a = torch.stack(s_lst), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.stack(s_prime_lst), \
                                          torch.tensor(done_lst, dtype=torch.float32), torch.tensor(prob_a_lst)
        W = torch.stack(W_lst)
        reachable_nodes = torch.stack(reachable_nodes_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, W, reachable_nodes
        
    def train_net(self, writer, it):
        s, a, r, s_prime, done_mask, prob_a, W, reachable_nodes = self.make_batch()
        a=a.to(device)
        r=r.to(device)
        done_mask=done_mask.to(device)
        prob_a=prob_a.to(device)

        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime, W, reachable_nodes) * done_mask
            delta = td_target - self.v(s, W, reachable_nodes)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float32).to(device)

            log_pi = self.pi(s, W, reachable_nodes)
            log_pi_a = log_pi.gather(1,a)
            ratio = torch.exp(log_pi_a - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s, W, reachable_nodes) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            
            parameters = self.theta1a.parameters()
            parameters_norm=torch.norm(torch.stack([torch.norm(p.grad.detach(),2) for p in parameters]), 2)
            writer.add_scalar('theta1a',parameters_norm,it)
            
            
            self.optimizer.step()
        return loss.mean().detach().cpu()
        
def GetTrainedModel(env, config, logdir='runs/test'):
    num_epi=config['num_epi']

    writer = SummaryWriter(logdir)
    model = PPONet(config).to(device)
    #writer.add_graph(model)
    score = 0.
    lossadd = 0.
    print_interval = config['print_interval']
    plot_interval=config['plot_interval']

    for n_epi in range(num_epi):
        env.reset()
        s = env.nfm
        done = False
        while not done:
            prob_logits = model.pi(s.unsqueeze(0), env.sp.W.unsqueeze(0), env.sp.W[env.state[0]].unsqueeze(0))
            pl_select = prob_logits > -torch.inf
            m = Categorical(logits = prob_logits[pl_select].squeeze())
            a = m.sample().item()

            _, r, done, info = env.step(a)
            s_prime = env.nfm

            model.put_data(( s, a, r/10., s_prime, m.probs[a].item(), done, env.sp.W, env.sp.W[env.state[0]]))
            s = s_prime

            score += r

        loss = model.train_net(writer, n_epi)
        lossadd+=loss

        if n_epi%plot_interval==0 and n_epi!=0:
            env.render_eupaths(fname='./'+logdir+'/rendering_active',t_suffix=False)
        if n_epi%print_interval==0 and n_epi!=0:
            p1,p2 = get_test_probs(env,model)
            

            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            print(p1)
            print(p2)
            writer.add_scalar('return per epi',score/print_interval,n_epi)
            writer.add_scalar('loss',loss/print_interval,n_epi)
            score = 0.
            lossadd=0.

    env.close()

def get_test_probs(env, model):
    env.reset()
    done=False
    probs=[]
    consecutive_action_queries=[3,4]
    for a in consecutive_action_queries:
        s = env.nfm
        with torch.no_grad():
            model.eval()
            prob_logits = model.pi(s.unsqueeze(0), env.sp.W.unsqueeze(0), env.sp.W[env.state[0]].unsqueeze(0))
            model.train()
            pl_select = prob_logits > -torch.inf
            m = Categorical(logits = prob_logits[pl_select].squeeze())
        probs.append(m.probs.detach().cpu())
        _,_,done,_ = env.step(a)
    return tuple(probs)


if __name__ == '__main__':
    #world_name='MetroU3_e17tborder_FixedEscapeInit'
    world_name='Manhattan3x3_WalkAround'
    
    #scenario_name='TrainMetro'
    state_repr='etUte0U0'
    state_enc='nfm'
    nfm_funcs = {
    'NFM_ev_ec_t'       : NFM_ev_ec_t(),
    'NFM_ec_t'          : NFM_ec_t(),
    'NFM_ev_t'          : NFM_ev_t(),
    'NFM_ev_ec_t_u'     : NFM_ev_ec_t_u(),
    'NFM_ev_ec_t_um_us' : NFM_ev_ec_t_um_us(),
    }
    edge_blocking = True
    nfm_func_name = 'NFM_ev_ec_t'
    nfm_func=nfm_funcs[nfm_func_name]
    remove_world_pool = True

    env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    env.redefine_nfm(nfm_func)
    env.capture_on_edges = edge_blocking

    # make task easier
    if remove_world_pool:
        env._remove_world_pool()
    #env.redefine_goal_nodes([7])
    #env.current_entry=1
    #SimulateInteractiveMode(env)
    
    #Hyperparameters
    config={}
    config['emb_dim']       = 16
    config['emb_iter_T']    = 3
    config['node_dim']      = env.nfm.shape[1]
    config['learning_rate'] = 0.0004
    config['gamma']         = 0.98
    config['lmbda']         = 0.95
    config['eps_clip']      = 0.05
    config['K_epoch']       = 5#3
    config['num_epi']       = 10000
    config['print_interval']= 20
    config['plot_interval'] = 100
    config['seed']          = 0
    seed_everything(config['seed'])

    logdir='runs/'+world_name+'_wpremoved'+str(remove_world_pool)+'/'+nfm_func_name+'_emb'+str(config['emb_dim'])+'_itT'+str(config['emb_iter_T'])+'_lr'+str(config['learning_rate'])+'_Kepoch'+str(config['K_epoch'])+'_qmimic_maxreach_2L/SEED'+str(config['seed']) 
    model = GetTrainedModel(env, config, logdir)