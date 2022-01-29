import copy
import warnings
warnings.filterwarnings("ignore")
# Demo of State2vec + DQN to solve a single world
import tqdm
from tkinter import W
import matplotlib.pyplot as plt
from modules.dqn.dqn_utils import seed_everything
from modules.gnn.comb_opt import QNet, QFunction, init_model, checkpoint_model, Memory
from modules.rl.rl_policy import GNN_s2v_Policy
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.rl_utils import EvaluatePolicy, EvalArgs2, GetFullCoverageSample
from modules.sim.simdata_utils import SimulateInteractiveMode
from modules.sim.graph_factory import GetPartialGraphEnvironments_Manh3x3
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from collections import namedtuple
Experience = namedtuple('Experience', ( \
    'state', 'state_tsr', 'W', 'action', 'action_nodeselect', 'reward', 'done', 'next_state', 'next_state_tsr'))

state_repr='etUt'
state_enc='nodes'
env_all = GetPartialGraphEnvironments_Manh3x3(state_repr=state_repr, state_enc=state_enc, edge_removals=[4,5,6], U=2, solvable=True, reachable_for_units=True)
#env_all = [env_all[1]]
#env_all[0].world_pool=[env_all[0].world_pool[4]]

config={}
config['node_dim']      = env_all[0].F
config['num_nodes']     = env_all[0].sp.V
config['emb_dim']       = 128        #128
config['emb_iter_T']    = 2         #2
#config['num_extra_layers']=0        #0
config['num_episodes']  = 5000       #500
config['memory_size']   = 1000      #200
config['num_step_ql']   = 1         #1
config['bsize']         = 64        #64
config['gamma']         = .9        #.9
config['lr_init']       = 1e-4      #1e-3
config['lr_decay']      = 0.99999    #0.99999
config['tau']           = 400       #inf                    # num grad steps for each target network update
config['eps_0']         = 1.        #1.
config['eps_min']       = 0.05#01       #0.05
#config['eps_decay']     = 0.##     #
epi_min=.9 # reach eps_min at % of episodes # .9
config['eps_decay']     = 1 - np.exp(np.log(config['eps_min'])/(epi_min*config['num_episodes']))
config['logdir']        = './results_Phase2/CombOpt'


def Test(config):
    qnet=QNet(config).to(device)
    xv=torch.tensor(env_all[0].nfm, dtype=torch.float32, device=device).unsqueeze(0)
    W=torch.tensor(env_all[0].sp.W,dtype=torch.float32,device=device).unsqueeze(0)
    y=qnet(xv,W)
    print(y)

def evaluate(affix, info=False):
    #Test(config)
    
    Q_func, Q_net, optimizer, lr_scheduler = init_model(config,fname='results_Phase2/CombOpt/'+affix+'/best_model.tar')
    #Q_func, Q_net, optimizer, lr_scheduler = init_model(config,fname='results_Phase2/CombOpt/'+affix+'/ep_350_length_7.0.tar')
    policy=GNN_s2v_Policy(Q_func)
    #policy.epsilon=0.
    #e=415
    #EvaluatePolicy(env_all[e], policy,env_all[e].world_pool, print_runs=False, save_plots=True, logdir='results_Phase2/CombOpt', eval_arg_func=EvalArgs2, silent_mode=False)
    R=[]

    
    for i,env in enumerate(tqdm.tqdm(env_all)):
        l, returns, c = EvaluatePolicy(env, policy,env.world_pool, print_runs=False, save_plots=False, logdir='results_Phase2/CombOpt/'+affix, eval_arg_func=EvalArgs2, silent_mode=True)
        if i%10==0:
            plotlist = GetFullCoverageSample(returns, env.world_pool, bins=3, n=3)
            EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir='results_Phase2/CombOpt/'+affix+'/runs/env'+str(env.hashint)+'_'+str(len(env.world_pool))+'entries', eval_arg_func=EvalArgs2, silent_mode=False)
        R+=returns 
    
    OF = open('results_Phase2/CombOpt/'+affix+'/Full_result.txt', 'w')
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

def train(seeds=1, seednr0=42):
    # Storing metrics about training:
    found_solutions = dict()  # episode --> (W, solution)
    losses = []
    path_length_ratios = []
    total_rewards=[]
    grad_update_count=0
    for seed in range(seeds):
        seed_everything(seed+seednr0) # 
        # Create module, optimizer, LR scheduler, and Q-function
        Q_func, Q_net, optimizer, lr_scheduler = init_model(config)
        Q_func_target, _, _, _ = init_model(config)
        #Q_func_target=Q_func

        logdir=config['logdir']+'/SEED'+str(seed+seednr0)
        writer=writer = SummaryWriter(log_dir=logdir)

        # Create memory
        memory = Memory(config['memory_size'])

        # keep track of mean ratio of estimated MVC / real MVC
        current_max_R = float('-inf')
        N_STEP_QL = config['num_step_ql']

        for episode in range(config['num_episodes']):
            # sample a new graph
            # current state (tuple and tensor)
            env=random.choice(env_all)
            current_state = env.reset()
            done=False   
            current_state_tsr = torch.tensor(env.nfm, dtype=torch.float32, device=device) 
            # Note: nfm = Graph Feature Matrix (FxV), columns are the node features, managed by the environment
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
                    action, action_nodeselect, _ = Q_func.get_best_action(current_state_tsr.to(dtype=torch.float32), env.sp.W, reachable_nodes)
                    if episode % 50 == 0:
                        pass
                        #print('Ep {} exploit | current sol: {} / next est reward: {} | sol: {}'.format(episode, solution, est_reward,solutions),'nextnode',next_node)
                
                next_state, reward, done, info = env.step(action)
                next_state_tsr = torch.tensor(env.nfm, dtype=torch.float32, device=device)
                
                # store rewards and states obtained along this episode:
                states.append(next_state)
                states_tsrs.append(next_state_tsr)
                rewards.append(reward)
                dones.append(done)
                actions.append(action)
                actions_nodeselect.append(action_nodeselect)
                
                # store our experience in memory, using n-step Q-learning:
                if len(actions) >= N_STEP_QL:
                    memory.remember(Experience(state          = states[-(N_STEP_QL+1)],
                                            state_tsr      = states_tsrs[-(N_STEP_QL+1)],
                                            W              = env.sp.W,
                                            action         = actions[-N_STEP_QL],
                                            action_nodeselect=actions_nodeselect[-N_STEP_QL],
                                            done           = dones[-N_STEP_QL], # CHECK!
                                            reward         = sum(rewards[-N_STEP_QL:]),
                                            next_state     = next_state,
                                            next_state_tsr = next_state_tsr))
                    
                if done:
                    for n in range(1, N_STEP_QL+1):
                        memory.remember(Experience(state=states[-(n+1)],
                                                state_tsr=states_tsrs[-(n+1)],
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
                if len(memory) >= config['bsize']:
                    experiences = memory.sample_batch(config['bsize'])
                    
                    batch_states_tsrs = [e.state_tsr for e in experiences]
                    batch_Ws = [torch.tensor(e.W,dtype=torch.float32,device=device) for e in experiences]
                    batch_actions = [e.action_nodeselect for e in experiences] #CHECK!
                    batch_targets = []

#                   Q_target=copy.deepcopy(policy.model)
#                   Q_target.load_state_dict(policy.model.state_dict())
     

                    for i, experience in enumerate(experiences):
                        target = experience.reward
                        if not experience.done:
                            with torch.no_grad():
                                _, _, best_reward = Q_func_target.get_best_action(experience.next_state_tsr, 
                                                                        experience.W,
                                                                        env.neighbors[experience.next_state[0]])
                            target += config['gamma'] * best_reward
                        batch_targets.append(target)
                        
                    # print('batch targets: {}'.format(batch_targets))
                    loss = Q_func.batch_update(batch_states_tsrs, batch_Ws, batch_actions, batch_targets)
                    grad_update_count+=1
                    losses.append(loss)
                    if grad_update_count % config['tau'] == 0:
                        #Q_func_target.model.load_state_dict(torch.load(Q_func.model.state.dict()))
                        Q_func_target.model = copy.deepcopy(Q_func.model)
                        print('Target network updated, epi=',episode,'grad_update_count=',grad_update_count)



            total_rewards.append(np.sum(rewards))
            """ Save model when we reach a new low average path length
            """
            #med_length = np.median(path_length_ratios[-100:])
            mean_R = int(np.mean(total_rewards[-10:])*10)/10
            if mean_R >= current_max_R:
                save_best_only = (mean_R == current_max_R)
                current_max_R = mean_R
                checkpoint_model(Q_net, optimizer, lr_scheduler, loss, episode, mean_R, logdir, best_only=save_best_only)

            writer.add_scalar("1a. epsilon", epsilon, episode)
            writer.add_scalar("1b. lr", optimizer.param_groups[0]['lr'], episode)
            writer.add_scalar("2. loss",    0 if loss is None else loss, episode)
            writer.add_scalar("3. epi_len", 1+len(actions_nodeselect), episode)
            writer.add_scalar("4. Reward per epi", total_rewards[-1], episode)


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
        plt.savefig(logdir+'/lossplot.png')
        plt.clf()

        plt.figure(figsize=(8,5))
        plt.plot(_moving_avg(total_rewards, 10))
        plt.title('Ratio (moving average) of (estimated MVC) / (real MVC)')
        plt.ylabel('ratio')
        plt.xlabel('episode')
        plt.savefig(logdir+'/ratioplot.png')
        plt.clf()

numseeds=1
seed0=1000
train(seeds=numseeds,seednr0=seed0)
evaluate('SEED'+str(seed0))