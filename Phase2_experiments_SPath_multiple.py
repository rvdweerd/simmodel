import copy
import warnings
warnings.filterwarnings("ignore")
# Demo of State2vec + DQN to solve a single world
import tqdm
from tkinter import W
import matplotlib.pyplot as plt
from modules.dqn.dqn_utils import seed_everything
from modules.gnn.comb_opt import QNet, QFunction, init_model, checkpoint_model, Memory
from modules.rl.rl_policy import GNN_s2v_Policy, ShortestPathPolicy, EpsilonGreedyPolicy
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.rl_algorithms import q_learning_exhaustive
from modules.rl.rl_utils import EvaluatePolicy, EvalArgs1, EvalArgs2, GetFullCoverageSample
from modules.sim.simdata_utils import SimulateInteractiveMode
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us
#from modules.sim.graph_factory import GetPartialGraphEnvironments_Manh3x3
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from collections import namedtuple
Experience = namedtuple('Experience', ( \
    'state', 'state_tsr', 'W', 'action', 'action_nodeselect', 'reward', 'done', 'next_state', 'next_state_tsr'))


def Test(config):
    qnet=QNet(config).to(device)
    xv=torch.tensor(env_all[0].nfm, dtype=torch.float32, device=device).unsqueeze(0)
    W=torch.tensor(env_all[0].sp.W,dtype=torch.float32,device=device).unsqueeze(0)
    y=qnet(xv,W)
    print(y)

def evaluate_tabular(logdir, env_all):
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

def evaluate_spath_heuristic(logdir, env_all):
    R=[]
    for i,env in enumerate(tqdm.tqdm(env_all)):
        policy=ShortestPathPolicy(env,weights='equal')
        l, returns, c = EvaluatePolicy(env, policy,env.world_pool, print_runs=False, save_plots=False, logdir=logdir, eval_arg_func=EvalArgs1, silent_mode=True)
        num_worlds_requested = 10
        once_every = max(1,len(env_all)//num_worlds_requested)
        if i % once_every ==0:
            plotlist = GetFullCoverageSample(returns, env.world_pool, bins=10, n=15)
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

def evaluate(logdir, info=False, config=None, env_all=None):
    #Test(config)
    
    Q_func, Q_net, optimizer, lr_scheduler = init_model(config,fname=logdir+'/best_model.tar')
    #Q_func, Q_net, optimizer, lr_scheduler = init_model(config,fname='results_Phase2/CombOpt/'+affix+'/ep_350_length_7.0.tar')
    policy=GNN_s2v_Policy(Q_func)
    #policy.epsilon=0.
    #e=415
    #EvaluatePolicy(env_all[e], policy,env_all[e].world_pool, print_runs=False, save_plots=True, logdir='results_Phase2/CombOpt', eval_arg_func=EvalArgs2, silent_mode=False)
    R=[]

    
    for i,env in enumerate(tqdm.tqdm(env_all)):
        l, returns, c = EvaluatePolicy(env, policy,env.world_pool, print_runs=False, save_plots=False, logdir=logdir, eval_arg_func=EvalArgs2, silent_mode=True)
        num_worlds_requested = 10
        once_every = max(1,len(env_all)//num_worlds_requested)
        if i % once_every ==0:
            plotlist = GetFullCoverageSample(returns, env.world_pool, bins=10, n=15)
            EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir=logdir, eval_arg_func=EvalArgs2, silent_mode=False, plot_each_timestep=False)
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

def train(seeds=1, seednr0=42, config=None, env_all=None):
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
        current_min_Ratio = float('+inf')
        current_max_Return= float('-inf')
        N_STEP_QL = config['num_step_ql']

        for episode in range(config['num_episodes']):
            # sample a new graph
            # current state (tuple and tensor)
            env=random.choice(env_all)
            #env=env_all[-1]
            env.reset()
            current_state = env.state
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
                
                _, reward, done, info = env.step(action)
                next_state = env.state
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



            success_ratio = len(actions_nodeselect) / env.sp.spath_length
            path_length_ratios.append(success_ratio)
            total_rewards.append(np.sum(rewards))

            """ Save model when we reach a new low average path length
            """
            #med_length = np.median(path_length_ratios[-100:])
            if (len(total_rewards)+1) % 5 == 0: # check every 5 episodes
                if config['optim_target']=='returns': # we seek to maximize the returns per episode
                    mean_Return = int(np.mean(total_rewards[-10:])*100)/100
                    if mean_Return >= current_max_Return:
                        save_best_only = (mean_Return == current_max_Return)
                        current_max_Return = mean_Return
                        checkpoint_model(Q_net, optimizer, lr_scheduler, loss, episode, mean_Return, logdir, best_only=save_best_only)                
                elif config['optim_target']=='ratios': # we seek to minimize the path lenths per episode w.r.t. shortest path to nearest target
                    mean_Ratio = int(np.mean(path_length_ratios[-10:])*100)/100
                    if mean_Ratio <= current_min_Ratio:
                        save_best_only = (mean_Ratio == current_min_Ratio)
                        current_min_Ratio = mean_Ratio
                        checkpoint_model(Q_net, optimizer, lr_scheduler, loss, episode, mean_Ratio, logdir, best_only=save_best_only)
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

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    

    # Model hyperparameters
    parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('--emb_itT', default=2, type=int)
    parser.add_argument('--num_epi', default=250, type=int)
    parser.add_argument('--mem_size', default=2000, type=int)
    parser.add_argument('--nfm_func', default='NFM_ev_ec_t', type=str)
    parser.add_argument('--scenario', default='None', type=str)
    parser.add_argument('--optim_target', default='None', type=str)
    parser.add_argument('--tau', default=100, type=int)
    
    args=parser.parse_args()

    world_name='SparseManhattan5x5'
    state_repr='etUt'
    state_enc='nfm'
    scenario_name=args.scenario

    env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    nfm_funcs = {'NFM_ev_ec_t':NFM_ev_ec_t(),'NFM_ec_t':NFM_ec_t(),'NFM_ev_t':NFM_ev_t(),'NFM_ev_ec_t_um_us':NFM_ev_ec_t_um_us()}
    nfm_func = nfm_funcs[args.nfm_func]
    env.redefine_nfm(nfm_func)
    env_all=[]
    if scenario_name=='1target-fixed24':
        env._remove_world_pool()
        env.databank={}
        env.register={}
        env.redefine_goal_nodes([24])
        env.current_entry=1
        env_all.append(copy.deepcopy(env))
        #SimulateInteractiveMode(env)
    elif scenario_name=='1target-random':
        env._remove_world_pool()
        env.databank={}
        env.register={}
        for i in range(env.sp.V):
            if i == env.sp.coord2labels[env.sp.start_escape_route]: 
                continue
            env.redefine_goal_nodes([i])
            env.current_entry=i
            env_all.append(copy.deepcopy(env))
    elif scenario_name=='2target-random':
        env._remove_world_pool()
        env.databank={}
        env.register={}        
        for i in range(env.sp.V):
            for j in range(0, i):
                if i == env.sp.coord2labels[env.sp.start_escape_route] or j == env.sp.coord2labels[env.sp.start_escape_route]:
                    continue
                if i==j:
                    assert False
                env.redefine_goal_nodes([i,j])
                env.current_entry=i
                env_all.append(copy.deepcopy(env))
    elif scenario_name == 'toptargets-fixed_3U-random-static':
        env.reset()
        #entry=env.current_entry
        #print('entry',entry)
        #SimulateInteractiveMode(env)#, entry=2200)
        # We clip the unit paths to the first position (no movement)
        for i in range(len(env.databank['coords'])):
            patharr=env.databank['coords'][i]['paths']
            for j in range(len(patharr)):
                patharr[j] = [patharr[j][0]]
            #env.databank['coords'][j]['paths']=patharr
            patharr=env.databank['labels'][i]['paths']
            for j in range(len(patharr)):
                patharr[j] = [patharr[j][0]]
            #env.databank['labels'][j]['paths']=patharr
        #SimulateInteractiveMode(env, entry=entry)
        env_all = [env]
    else:
        assert False

    config={}
    config['node_dim']      = env_all[0].F
    config['num_nodes']     = env_all[0].sp.V
    config['scenario_name'] = args.scenario
    config['nfm_func']      = args.nfm_func
    config['emb_dim']       = args.emb_dim #32        #128
    config['emb_iter_T']    = args.emb_itT #2
    config['optim_target']  = args.optim_target
    #config['num_extra_layers']=0        #0
    config['num_episodes']  = args.num_epi #2500       #500
    config['memory_size']   = args.mem_size #2000      #200
    config['num_step_ql']   = 1         #1
    config['bsize']         = 64        #64
    config['gamma']         = .9        #.9
    config['lr_init']       = 1e-3      #1e-3
    config['lr_decay']      = 0.99999    #0.99999
    config['tau']           = args.tau       #100                     # num grad steps for each target network update
    config['eps_0']         = 1.        #1.
    config['eps_min']       = 0.1#01       #0.05
    #config['eps_decay']     = 0.##     #
    epi_min=.9 # reach eps_min at % of episodes # .9
    config['eps_decay']     = 1 - np.exp(np.log(config['eps_min'])/(epi_min*config['num_episodes']))
    rootdir='./results_Phase2/SPath/'+ \
                                world_name+'/'+ \
                                scenario_name
    config['logdir']        = rootdir + '/' + \
                                nfm_func.name+'/'+ \
                                '_emb'+str(config['emb_dim']) + \
                                '_itT'+str(config['emb_iter_T']) + \
                                '_epi'+str(config['num_episodes']) + \
                                '_mem'+str(config['memory_size'])

    numseeds=1
    seed0=999999
    train(seeds=numseeds,seednr0=seed0, config=config, env_all=env_all)
    evaluate(logdir=config['logdir']+'/SEED'+str(seed0), config=config, env_all=env_all)
    #evaluate_spath_heuristic(logdir=rootdir+'/heur/spath', env_all=env_all)
    #evaluate_tabular(logdir=rootdir+'/tabular', env_all=env_all)