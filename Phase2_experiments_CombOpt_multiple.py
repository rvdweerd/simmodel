# Demo of State2vec + DQN to solve a single world
import tqdm
from tkinter import W
import matplotlib.pyplot as plt
from modules.dqn.dqn_utils import seed_everything
from modules.gnn.comb_opt import QNet, QFunction, init_model, checkpoint_model, Memory
from modules.rl.rl_policy import GNN_s2v_Policy
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.rl_utils import EvaluatePolicy, EvalArgs2
from modules.sim.simdata_utils import SimulateInteractiveMode
from modules.sim.graph_factory import GetPartialGraphEnvironments_Manh3x3
import random
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from collections import namedtuple
Experience = namedtuple('Experience', ( \
    'state', 'state_tsr', 'W', 'action', 'action_nodeselect', 'reward', 'done', 'next_state', 'next_state_tsr'))

state_repr='etUt'
state_enc='nodes'
env_all = GetPartialGraphEnvironments_Manh3x3(state_repr=state_repr, state_enc=state_enc, edge_removals=[4,5,6], U=2, solvable=True, reachable_for_units=True)

seed_everything(42) # other seeds require more steps (1000)
config={}
config['node_dim']      = env_all[0].F
config['num_nodes']     = env_all[0].sp.V
config['emb_dim']       = 32        #32
config['emb_iter_T']    = 3         #3
config['num_extra_layers']=1        #1
config['num_episodes']  = 2000       #250
config['memory_size']   = 1000       #200
config['num_step_ql']   = 1         #1
config['bsize']         = 64        #64
config['gamma']         = .9        #.9
config['lr_init']       = 1e-3      #1e-3
config['lr_decay']      = 0.999998  #0.999998
config['eps_0']         = 1.        #1.
config['eps_min']       = 0.1       #0.1
config['eps_decay']     = 0.002     #0.01
config['logdir']        = './results_Phase2/CombOpt2'

def Test(config):
    qnet=QNet(config).to(device)
    xv=torch.tensor(env_all[0].gfm, dtype=torch.float32, device=device).unsqueeze(0)
    W=torch.tensor(env_all[0].sp.W,dtype=torch.float32,device=device).unsqueeze(0)
    y=qnet(xv,W)
    print(y)

def evaluate():
    #Test(config)
    Q_func, Q_net, optimizer, lr_scheduler = init_model(config,fname='results_Phase2/CombOpt2/ep_1600_length_4.3.tar')
    policy=GNN_s2v_Policy(Q_func)
    #policy.epsilon=0.
    #e=415
    #EvaluatePolicy(env_all[e], policy,env_all[e].world_pool, print_runs=False, save_plots=True, logdir='results_Phase2/CombOpt2', eval_arg_func=EvalArgs2, silent_mode=False)
    R=[]
    for i,env in enumerate(tqdm.tqdm(env_all)):
        l,r,c = EvaluatePolicy(env, policy,env.world_pool, print_runs=False, save_plots=False, logdir='results_Phase2/CombOpt2', eval_arg_func=EvalArgs2, silent_mode=True)
        #print('env',i,'mean_r {:.1f}'.format(np.mean(r)),r)
        R+=r 
    print('Total instances evaluated:',len(R),'Avg reward',np.mean(R))
    possol=np.sum(np.array(R)>0)
    print('Number of >0 solutions:',possol,' ({:.1f}'.format(possol/len(R)*100)+'%)')

def train():
    # Create module, optimizer, LR scheduler, and Q-function
    Q_func, Q_net, optimizer, lr_scheduler = init_model(config)

    # Create memory
    memory = Memory(config['memory_size'])

    # Storing metrics about training:
    found_solutions = dict()  # episode --> (W, solution)
    losses = []
    path_length_ratios = []

    # keep track of mean ratio of estimated MVC / real MVC
    current_max_R = float('-inf')
    N_STEP_QL = config['num_step_ql']
    total_rewards=[]

    for episode in range(config['num_episodes']):
        # sample a new graph
        # current state (tuple and tensor)
        env=random.choice(env_all)
        current_state = env.reset()
        done=False   
        current_state_tsr = torch.tensor(env.gfm, dtype=torch.float32, device=device) 
        # Note: gfm = Graph Feature Matrix (FxV), columns are the node features, managed by the environment
        # It's currently defined (for each node) as:
        #   [.] node number
        #   [.] 1 if target node, 0 otherwise 
        #   [.] # of units present at node at current time

        # Keep track of some variables for insertion in replay memory:
        states = [current_state]
        states_tsrs = [current_state_tsr]  # we also keep the state tensors here (for efficiency)
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
                action_nodeselect = env.neighbors[env.state[0]][action]
                nr_explores += 1
                if episode % 50 == 0:
                    pass
                    #print('Ep {} explore | current sol: {} | sol: {}'.format(episode, solution, solutions),'nextnode',next_node)
            else:
                # exploit
                #with torch.no_grad(): #(already dealt with inside function)
                reachable_nodes=env.neighbors[env.state[0]]
                action, action_nodeselect, _ = Q_func.get_best_action(current_state_tsr, env.sp.W, reachable_nodes)
                if episode % 50 == 0:
                    pass
                    #print('Ep {} exploit | current sol: {} / next est reward: {} | sol: {}'.format(episode, solution, est_reward,solutions),'nextnode',next_node)
            
            next_state, reward, done, info = env.step(action)
            next_state_tsr = torch.tensor(env.gfm, dtype=torch.float32, device=device)
            
            # store rewards and states obtained along this episode:
            states.append(next_state)
            states_tsrs.append(next_state_tsr)
            rewards.append(reward)
            dones.append(done)
            actions.append(action)
            actions_nodeselect.append(action_nodeselect)
            
            # store our experience in memory, using n-step Q-learning:
            if len(actions) > N_STEP_QL:
                memory.remember(Experience(state          = states[-N_STEP_QL],
                                        state_tsr      = states_tsrs[-N_STEP_QL],
                                        W              = env.sp.W,
                                        action         = actions[-N_STEP_QL],
                                        action_nodeselect=actions_nodeselect[-N_STEP_QL],
                                        done           = dones[-N_STEP_QL], # CHECK!
                                        reward         = sum(rewards[-N_STEP_QL:]),
                                        next_state     = next_state,
                                        next_state_tsr = next_state_tsr))
                
            if done:
                for n in range(1, N_STEP_QL):
                    memory.remember(Experience(state=states[-n],
                                            state_tsr=states_tsrs[-n],
                                            W = env.sp.W, 
                                            action=actions[-n],
                                            action_nodeselect=actions_nodeselect[-n], 
                                            done=dones[-n],
                                            reward=sum(rewards[-n:]), 
                                            next_state=next_state,
                                            next_state_tsr=next_state_tsr))
            
            # update state and current solution
            current_state = next_state
            current_state_tsr = next_state_tsr
            
            # take a gradient step
            loss = None
            if len(memory) >= 3*config['bsize']:
                experiences = memory.sample_batch(config['bsize'])
                
                batch_states_tsrs = [e.state_tsr for e in experiences]
                batch_Ws = [torch.tensor(e.W,dtype=torch.float32,device=device) for e in experiences]
                batch_actions = [e.action_nodeselect for e in experiences] #CHECK!
                batch_targets = []
                
                for i, experience in enumerate(experiences):
                    target = experience.reward
                    if not experience.done:
                        _, _, best_reward = Q_func.get_best_action(experience.next_state_tsr, 
                                                                experience.W,
                                                                env.neighbors[experience.next_state[0]])
                        target += config['gamma'] * best_reward
                    batch_targets.append(target)
                    
                # print('batch targets: {}'.format(batch_targets))
                loss = Q_func.batch_update(batch_states_tsrs, batch_Ws, batch_actions, batch_targets)
                losses.append(loss)
                
                """ Save model when we reach a new low average path length
                """
                #med_length = np.median(path_length_ratios[-100:])
                mean_R = int(np.mean(total_rewards[-10:])*10)/10
                if mean_R > current_max_R:
                    current_max_R = mean_R
                    checkpoint_model(Q_net, optimizer, lr_scheduler, loss, episode, mean_R, config['logdir'])

        if np.sum(rewards)>0:
            k=0            
        total_rewards.append(np.sum(rewards))

        if episode % 10 == 0:
            print('Ep %d. Loss = %.3f / median R = %.3f / last R = %.4f / epsilon = %.4f / lr = %.4f / mem = %d / walk %s.' % (
                episode, (-1 if loss is None else loss), np.mean(total_rewards[-10:]), total_rewards[-1], epsilon,
                Q_func.optimizer.param_groups[0]['lr'],len(memory),str(actions_nodeselect)))
            #print(path_length_ratios[-50:])
            #found_solutions[episode] = (W.clone(), [n for n in solution])


    def _moving_avg(x, N=10):
        return np.convolve(np.array(x), np.ones((N,))/N, mode='valid')

    plt.figure(figsize=(8,5))
    plt.semilogy(_moving_avg(losses, 100))
    plt.title('Loss (moving average) during training')
    plt.ylabel('loss')
    plt.xlabel('training iteration')
    plt.savefig(config['logdir']+'/lossplot.png')
    plt.clf()

    plt.figure(figsize=(8,5))
    plt.plot(_moving_avg(total_rewards, 100))
    plt.title('Ratio (moving average) of (estimated MVC) / (real MVC)')
    plt.ylabel('ratio')
    plt.xlabel('episode')
    plt.savefig(config['logdir']+'/ratioplot.png')
    plt.clf()

evaluate()