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
num_iter    = 1*250#000
gamma       = .9
alpha_0     = .2
alpha_decay = 0.
initial_Q_values = 10.

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['Manhattan3']
conf['direction_north']=False
fixed_init=conf['fixed_initial_positions']

env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation='et')

env.register['coords']={((1,0),(0,2),(1,2),(2,2)):0}
env.register['labels']={(1,6,7,8):0}
env.databank['coords']=[{'start_escape_route':(1,0), 'start_units':[(0,2),(1,2),(2,2)], 'paths':[[(0,2),(0,1),(0,0),(0,1),(0,0)],[(1,2),(1,2),(1,2),(0,2),(0,1)],[(2,2),(2,1)]]}]
env.databank['labels']=[{'start_escape_route':1, 'start_units':[6,7,8], 'paths':[[6,3,0,3,0],[7,7,7,6,3],[8,5]]}]
env.iratios=[1.]
env.all_worlds=[0]
env.world_pool=[0]

policy = EpsilonGreedyPolicy(env, eps_0, initial_Q_values)

metrics_episode_returns = {}
metrics_episode_lengths = {}
metrics_avgperstep = {}
Q_tables = {}

algos  = [q_learning_exhaustive]#,sarsa,expected_sarsa]
for algo in algos:
    metrics_all = np.zeros((num_seeds,2,num_iter))
    for s in range(num_seeds):
        #seed_everthing(seed=s)
        policy.Reset()
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
EvaluatePolicy(env,policy,env.world_pool,print_runs=False, save_plots=True)
#env.fixed_initial_positions=None
