from environments import GraphWorld
from dqn_utils import seed_everything
from rl_policy import EpsilonGreedyPolicy
from rl_algorithms import q_learning, sarsa, expected_sarsa, q_learning_exhaustive
from rl_plotting import PlotPerformanceCharts, PlotGridValues, PlotNodeValues
from rl_utils import EvaluatePolicy, CreateDuplicatesTrainsets
from rl_custom_worlds import GetCustomWorld
import numpy as np
import simdata_utils as su
import random 

# Select graph world
#env=GetCustomWorld('Manhattan3x3_TestWorld')
#env=GetCustomWorld('Manhattan3x3_PauseFreezeWorld')
env=GetCustomWorld('Manhattan3x3_PauseDynamicWorld')
env._encode=env._encode_nodes
env.state_representation = 'etUte0U0'

# Select hyperparameters
num_seeds   = 1
eps_0       = .2
eps_min     = 0.2
cutoff      = 1#200
num_iter    = 1*250#000
gamma       = .9
alpha_0     = .2
alpha_decay = 0.
initial_Q_values = 10.

# Initialize
policy = EpsilonGreedyPolicy(env, eps_0, initial_Q_values)
metrics_episode_returns = {}
metrics_episode_lengths = {}
metrics_avgperstep = {}
Q_tables = {}

# Run Q-learning
algos  = [q_learning_exhaustive]#,sarsa,expected_sarsa]
for algo in algos:
    metrics_all = np.zeros((num_seeds,2,num_iter))
    for s in range(num_seeds):
        seed_everything(seed=s)
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
    print(k,v)
    for i in v:
        count+=1
print('Total number of q values stored',count)
EvaluatePolicy(env,policy,env.world_pool,print_runs=False, save_plots=True)
#env.fixed_initial_positions=None
