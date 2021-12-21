#from environments import GraphWorld
from dqn_utils import seed_everything
from rl_policy import EpsilonGreedyPolicy
from rl_algorithms import q_learning_exhaustive
#from rl_plotting import PlotPerformanceCharts, PlotGridValues, PlotNodeValues
from rl_utils import EvaluatePolicy, CreateDuplicatesTrainsets, GetOutliersSample, GetFullCoverageSample
from rl_custom_worlds import CreateWorlds
import numpy as np
#import simdata_utils as su
#import random 
import pickle

num_seeds   = 1
eps_0       = 1.
eps_min     = 0.1
#cutoff      = 1#200
num_iter    = 1000
gamma       = .9#.9
alpha_0     = .2
alpha_decay = 0.
initial_Q_values = 10.

run_world_names = [
    'Manhattan5x5_FixedEscapeInit',
    #'MetroU3_e17tborder_FixedEscapeInit',
    #'MetroU3_e17t31_FixedEscapeInit', 
    #'MetroU3_e17t0_FixedEscapeInit', 
    #'Manhattan5x5_VariableEscapeInit',        
]
worlds = CreateWorlds(run_world_names, make_reflexive=True, state_repr='etUt', state_enc='nodes')

for env, world_name in zip(worlds, run_world_names):
    policy = EpsilonGreedyPolicy(env, eps_0, eps_min, initial_Q_values)
    #init_pos_trainset_indices0, init_pos_trainset_indices1 = CreateDuplicatesTrainsets(env, min_y_coord=env.sp.N-1, min_num_same_positions=env.sp.U, min_num_worlds=4)
    #env.world_pool = init_pos_trainset_indices1 # limit the training set to the selected entries
    #env.world_pool = random.sample(env.all_worlds,10)
    env.world_pool = env.all_worlds[:11]

    metrics_episode_returns = {}
    metrics_episode_lengths = {}
    metrics_avgperstep = {}
    Q_tables = {}

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
    #PlotPerformanceCharts(algos, performance_metrics)
    #PlotNodeValues(algos,env,Q_tables)

    #import matplotlib.pyplot as plt
    #plt.clf()
    policy.epsilon=0.
    logdir = 'results/TabQL/'+world_name+'/'+env.state_representation
    _, returns, _ = EvaluatePolicy(env,policy,env.world_pool,print_runs=False, save_plots=False, logdir=logdir)
    #plotlist = GetOutliersSample(returns)
    plotlist = GetFullCoverageSample(returns,bins=10,n=10)
    EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir=logdir)
    saveinfo = {"Q":policy.Q}
    a_file = open(logdir + "/" + "Q_table.pkl", "wb")
    pickle.dump(saveinfo, a_file)
    a_file.close()    
