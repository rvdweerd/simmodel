from environments import GraphWorld
from rl_policy import EpsilonGreedyPolicy
from rl_algorithms import q_learning, sarsa, expected_sarsa
from rl_plotting import PlotPerformanceCharts, PlotGridValues, PlotNodeValues
from rl_utils import EvaluatePolicy, SelectTrainset
import numpy as np
import simdata_utils as su

num_seeds   = 10
eps_0       = .2
eps_min     = 0.
cutoff      = 100#200
num_iter    = 500
gamma       = 1.
alpha_0     = .5
alpha_decay = 0.
initial_Q_values = 0.

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['Manhattan5']
conf['direction_north']=False

#env = GraphWorld(conf, optimization_method='dynamic', fixed_initial_positions=(2,15,19,22),state_representation='ete0U0')
env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation='etUte0U0')

policy = EpsilonGreedyPolicy(env, eps_0, eps_min, cutoff, initial_Q_values)
init_pos_trainset_indices0, init_pos_trainset_indices1 = SelectTrainset(env, min_y_coord=env.sp.N-1, min_num_same_positions=env.sp.U, min_num_worlds=4)
env.world_pool = init_pos_trainset_indices1 # limit the training set to the selected entries

metrics_episode_returns = {}
metrics_episode_lengths = {}
metrics_avgperstep = {}
Q_tables = {}

algos  = [q_learning]#,sarsa,expected_sarsa]
for algo in algos:
    metrics_all = np.zeros((num_seeds,2,num_iter))
    for s in range(num_seeds):
        #seed_everthing(seed=s)
        policy.Reset()
        Q_table, metrics_singleseed, policy, _ = algo(env, policy, num_iter, discount_factor=gamma, alpha_0=alpha_0, alpha_decay=alpha_decay,print_episodes=False)
        metrics_all[s] = metrics_singleseed
    
    Q_tables[algo.__name__] = Q_table
    metrics_episode_returns[algo.__name__] = metrics_all[:, 0, :]
    metrics_episode_lengths[algo.__name__] = metrics_all[:, 1, :]
    metrics_avgperstep[algo.__name__] = np.sum(
        metrics_episode_returns[algo.__name__], axis=0)/np.sum(metrics_episode_lengths[algo.__name__], axis=0)
performance_metrics = { 'e_returns': metrics_episode_returns, 'e_lengths':metrics_episode_lengths, 'rps':metrics_avgperstep}

PlotPerformanceCharts(algos, performance_metrics)
PlotNodeValues(algos,env,Q_tables)
import matplotlib.pyplot as plt
plt.clf()
EvaluatePolicy(env,policy,env.world_pool,save_plots=True)
#env.fixed_initial_positions=None
