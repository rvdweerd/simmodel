from environments import GraphWorld
from rl_policy import EpsilonGreedyPolicy
from rl_algorithms import q_learning, sarsa, expected_sarsa, q_learning_exhaustive
from rl_plotting import PlotPerformanceCharts, PlotGridValues, PlotNodeValues
from rl_utils import EvaluatePolicy, CreateDuplicatesTrainsets
import numpy as np
import simdata_utils as su
import random 

num_seeds   = 1
eps_0       = .2
eps_min     = 0.2
cutoff      = 1#200
num_iter    = 1000*250#2600*250
gamma       = .9
alpha_0     = .2
alpha_decay = 0.
initial_Q_values = 10.

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
#conf=configs['Manhattan5']
conf=configs['MetroGraphU3']
conf['direction_north']=False

#env = GraphWorld(conf, optimization_method='dynamic', fixed_initial_positions=(2,15,19,22),state_representation='ete0U0')
env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation='etUt')

policy = EpsilonGreedyPolicy(env, eps_0, initial_Q_values)
#init_pos_trainset_indices0, init_pos_trainset_indices1 = CreateDuplicatesTrainsets(env, min_y_coord=env.sp.N-1, min_num_same_positions=env.sp.U, min_num_worlds=4)
#env.world_pool = init_pos_trainset_indices1 # limit the training set to the selected entries
#env.world_pool = [env.all_worlds[env.register['labels'][(2,4,5,22)]]]#random.sample(env.all_worlds,10)
#env.world_pool = random.sample(env.all_worlds,260)
env.world_pool = env.all_worlds

metrics_episode_returns = {}
metrics_episode_lengths = {}
metrics_avgperstep = {}
Q_tables = {}

algos  = [q_learning_exhaustive]#,sarsa,expected_sarsa]
for algo in algos:
    metrics_all = np.zeros((num_seeds,2,num_iter))
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
import matplotlib.pyplot as plt
plt.clf()
count=0
for k,v in policy.Q.items():
    for i in v:
        count+=1
print('Total number of q values stored',count)
EvaluatePolicy(env,policy,env.world_pool,print_runs=False, save_plots=False)
#env.fixed_initial_positions=None
