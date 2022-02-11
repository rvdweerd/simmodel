import copy
import numpy as np
from gym import ObservationWrapper, ActionWrapper
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
class PPO_ActWrapper(ActionWrapper):
    """Wrapper for processing actions defined as next node label."""

    def __init__(self, env):
        super().__init__(env)
        
    def action(self, action):
        """convert action."""
        assert action in self.neighbors[self.state[0]]
        a= self.neighbors[self.state[0]].index(action)
        #print('Node_select action:',action,'Neighbor_index action:',a)
        return a

# code from https://github.com/seungeunrho/minimalRL/blob/master/ppo.py
class Struc2Vec(nn.Module):
    def __init__(self, emb_dim, emb_iter_T, node_dim):
        super().__init__()
        self.emb_dim        = emb_dim
        self.T              = emb_iter_T
        self.node_dim       = node_dim

        self.theta1a    = nn.Linear(self.node_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta1b    = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta2     = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta3     = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta4     = nn.Linear(1, self.emb_dim, True)#, dtype=torch.float32)

    def forward(self, xv, Ws):
        # xv: The node features (batch_size, num_nodes, node_dim)
        # Ws: The graphs (batch_size, num_nodes, num_nodes)
        xv=xv.to(device)
        Ws=Ws.to(device)
        
        
        num_nodes = xv.shape[1]
        batch_size = xv.shape[0]
        
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
        return mu

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

        self.s2v = Struc2Vec(self.emb_dim, self.T, self.node_dim)
        
        self.theta5_pi1 = nn.Linear(2*self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta5_pi2 = nn.Linear(self.emb_dim, 1, True, dtype=torch.float32)
        self.theta6_pi  = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta7_pi  = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)

        #self.s2v_v = Struc2Vec(self.emb_dim, self.T, self.node_dim)
        self.theta5_v1 = nn.Linear(2*self.emb_dim, self.emb_dim, True, dtype=torch.float32)
        self.theta5_v2 = nn.Linear(self.emb_dim, 1, True, dtype=torch.float32)
        self.theta6_v  = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta7_v  = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.numTrainableParameters()

    def pi(self, x, W, reachable_nodes):
        # reachable_nodes (batch_dim, nr_nodes)
        reachable_nodes=reachable_nodes.type(torch.BoolTensor)

        mu = self.s2v(x, W) # (batch_dim, nr_nodes, 2*emb_dim)
        num_nodes=x.shape[1]
        global_state = self.theta6_pi(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1))
        local_action = self.theta7_pi(mu)  # (batch_dim, nr_nodes, emb_dim)
        rep= F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (batch_dim, nr_nodes, 2*emb_dim)


        rep = self.theta5_pi1(F.relu(rep))
        #rep = F.relu(self.theta5_pi1(rep))
        prob_logits = ((self.theta5_pi2(rep))).squeeze(dim=2) # (batch_dim, nr_nodes)
        prob_logits[~reachable_nodes] = -torch.inf
        #prob_vals = prob_vals[reachable_nodes]
        
        # mask invalid actions
        #print(self.count,prob_logits)
        #self.count+=1
        return prob_logits # returns the logits (batch_dim, nr_nodes)

    
    def v(self, x, W, reachable_nodes, snode):
        mu = self.s2v(x, W) # (batch_dim, nr_nodes, 2*emb_dim)
        num_nodes=x.shape[1]
        global_state = self.theta6_v(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1))
        local_action = self.theta7_v(mu)  # (batch_dim, nr_nodes, emb_dim)
        rep= F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (batch_dim, nr_nodes, 2*emb_dim)
        
        # # OPTION 1: aggregate (mean = best) and project
        # ##rep2 = torch.max(rep,dim=1)[0]
        # rep = rep.mean(dim=1)
        #rep = rep[[i for i in range(rep.shape[0])],snode,:]  # select embeddings of escaper nodes -> (bsize,emb_dim)  
        
        # OPTION 2: mimic Q-function and take the max in the reachable node dimension
        ###rep = F.relu(self.theta5_v1(rep))
        rep = self.theta5_v1(F.relu(rep))
        qvals = self.theta5_v2(rep).squeeze(-1) # (batch_dim, nr_nodes)
        reachable_nodes=reachable_nodes.type(torch.BoolTensor).to(device)
        qvals[~reachable_nodes] = -torch.inf
        v=torch.max(qvals,dim=1)[0]
        #v = torch.gather(rep,1,snode[:,None])
        
        # optional: ONLY LOOK AT DIRECT NEIGHBORHOOD
        #v = (rep * reachable_nodes).mean(dim=1)
        #v = rep.mean(dim=1)
        return v # (batch_dim): a value of the current graph state for each transition
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst, W_lst, reachable_nodes_lst, snode_lst = [], [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done ,W, reachable_nodes, snode = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_mask = 0 if done else 1
            done_lst.append(done_mask)
            W_lst.append(W)
            reachable_nodes_lst.append(reachable_nodes)
            snode_lst.append(snode)
            
        s,a,r,s_prime,done_mask, prob_log_a = torch.stack(s_lst), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.stack(s_prime_lst), \
                                          torch.tensor(done_lst, dtype=torch.float32), torch.tensor(prob_a_lst)
        W = torch.stack(W_lst)
        reachable_nodes = torch.stack(reachable_nodes_lst)
        snode = torch.tensor(snode_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_log_a, W, reachable_nodes, snode
        
    def train_net(self, writer, it):
        s, a, r, s_prime, done_mask, prob_log_a, W, reachable_nodes, snode = self.make_batch()
        a=a.to(device)
        r=r.to(device)
        done_mask=done_mask.to(device)
        prob_log_a=prob_log_a.to(device)
        snode=snode.to(device)

        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime, W, reachable_nodes, snode) * done_mask
            delta = td_target - self.v(s, W, reachable_nodes, snode)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t
                advantage_lst.append(advantage)
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float32).to(device)

            log_pi = self.pi(s, W, reachable_nodes)
            m = Categorical(logits = log_pi)
            
            log_pi_a = m.log_prob(a.squeeze())
            ratio = torch.exp(log_pi_a - prob_log_a)  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s, W, reachable_nodes, snode) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            
            if it%100==0:
                parameters = self.s2v.parameters()
                parameters_norm=torch.norm(torch.stack([torch.norm(p.grad.detach(),2) for p in parameters]), 2)
                writer.add_scalar('s2v_pi grad norm',parameters_norm,it)

                # parameters = self.theta5_pi.parameters()
                # parameters_norm=torch.norm(torch.stack([torch.norm(p.grad.detach(),2) for p in parameters]), 2)
                # writer.add_scalar('theta5_pi grad norm',parameters_norm,it)

                # parameters = self.theta5_v.parameters()
                # parameters_norm=torch.norm(torch.stack([torch.norm(p.grad.detach(),2) for p in parameters]), 2)
                # writer.add_scalar('theta5_v grad norm',parameters_norm,it)
            
            
            self.optimizer.step()
        return loss.mean().detach().cpu()

    def numTrainableParameters(self):
        print('PPO model (pi and v) size:')
        print('------------------------------------------')
        total = 0
        for name, p in self.named_parameters():
            total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("Total number of parameters: {}".format(total))
        print('------------------------------------------')
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

        
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
        scores=[]
        entries=[]
        for i in range(5):
            env.reset()
            s = copy.deepcopy(env.nfm)
            done = False
            score_local=0
            epientry=[]
            while not done:
                with torch.no_grad(): #TODO check
                    prob_logits = model.pi(s.unsqueeze(0), env.sp.W.unsqueeze(0), env.sp.W[env.state[0]].unsqueeze(0))
                #pl_select = prob_logits > -torch.inf
                #prob_logits = prob_logits[prob_logits>-torch.inf]
                m = Categorical(logits = prob_logits.squeeze())
                a = m.sample()
                a_logprob = m.log_prob(a)
                snode = env.state[0]

                _, r, done, info = env.step(a)
                s_prime = copy.deepcopy(env.nfm)

                entry=( s, a.detach().item(), r/10, s_prime, a_logprob.detach().item(), done, env.sp.W, env.sp.W[snode], snode)
                epientry.append(entry)
                #model.put_data(( s, a.detach().item(), r/10, s_prime, a_logprob.detach().item(), done, env.sp.W, env.sp.W[snode], snode))
                s = copy.deepcopy(s_prime)

                score += r
                score_local+=r
            scores.append(score_local)
            entries.append(epientry)
        maxindex=np.argmax(np.array(scores))
        if scores[maxindex] > 0:
            model.K_epoch=10
        else:
            model.K_epoch=1
        for e in entries[maxindex]:
            model.put_data(e)

        loss = model.train_net(writer, n_epi)
        lossadd+=loss

        if n_epi%plot_interval==0 and n_epi!=0:
            env.render_eupaths(fname='./'+logdir+'/rendering_active',t_suffix=False)
        if n_epi%print_interval==0 and n_epi!=0:
            test_probs = get_test_probs(env,model)
            
            np.set_printoptions(formatter={'float':"{0:0.1f}".format})
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            evalstates=sorted(list(test_probs.keys()))
            for s in evalstates:
                print('state:',s,'probs: ',end='')
                for pr in test_probs[s]:
                    if pr>0.01:
                        print('{:.2f}'.format(pr)+'  ',end='')
                    else:
                        print('  .   ',end='')
                print(' - best option', np.argmax(test_probs[s]))
            writer.add_scalar('return per epi',score/print_interval,n_epi)
            writer.add_scalar('loss',loss/print_interval,n_epi)
            score = 0.
            lossadd=0.

    env.close()

def get_test_probs(env, model):
    env.reset()
    done=False
    probs={}
    t=env.sp.T
    env.sp.T=100
    consecutive_action_queries=[2,5,8,7]
    for a in consecutive_action_queries:
        s = env.nfm
        with torch.no_grad():
            model.eval()
            prob_logits = model.pi(s.unsqueeze(0), env.sp.W.unsqueeze(0), env.sp.W[env.state[0]].unsqueeze(0))
            model.train()
            #pl_select = prob_logits > -torch.inf
            m = Categorical(logits = prob_logits.squeeze())
        probs[env.state[0]] = m.probs.detach().cpu().numpy()
        _,_,done,_ = env.step(a)
    env.sp.T=t
    return probs


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
    nfm_func_name = 'NFM_ev_ec_t'#_um_us'
    nfm_func=nfm_funcs[nfm_func_name]
    remove_world_pool = False
    env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    env.redefine_nfm(nfm_func)
    env.redefine_goal_nodes([7])
    env.capture_on_edges = edge_blocking

    # make task easier
    if remove_world_pool:
        env._remove_world_pool()
    #env.redefine_goal_nodes([7])
    #env.current_entry=1
    SimulateInteractiveMode(env)
    
    env = PPO_ActWrapper(env)

    #Hyperparameters
    config={}
    config['emb_dim']       = 32
    config['emb_iter_T']    = 5
    config['node_dim']      = env.nfm.shape[1]
    config['learning_rate'] = 0.0001
    config['gamma']         = 0.98
    config['lmbda']         = 0.95
    config['eps_clip']      = 0.05
    config['K_epoch']       = 1#3
    config['num_epi']       = 10000
    config['print_interval']= 20
    config['plot_interval'] = 100
    #config['seed']          = 10

    for seed in [0,1,2,3,4]:
        seed_everything(seed)
        logdir='results/gnn-ppo/runs/'+world_name+'_wpremoved'+str(remove_world_pool)+'/'+nfm_func_name+'_emb'+str(config['emb_dim'])+'_itT'+str(config['emb_iter_T'])+'_lr'+str(config['learning_rate'])+'_Kepoch'+str(config['K_epoch'])+'_qmimic_maxreach_2L/SEED'+str(seed) 
        model = GetTrainedModel(env, config, logdir)