from environments import GraphWorld
from rl_policy import EpsilonGreedyPolicy
from rl_algorithms import q_learning, sarsa, expected_sarsa
from rl_plotting import PlotPerformanceCharts, PlotGridValues, PlotNodeValues
from rl_utils import EvaluatePolicy
import numpy as np
import simdata_utils as su

num_seeds   = 1
eps_0       = 1.0
eps_min     = 0.
cutoff      = 500
num_iter    = 1500
gamma       = 1.
alpha_0     = .1
alpha_decay = 0.
initial_Q_values = 10.0

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['Manhattan5']
conf['direction_north']=False

env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=((2,0),(0,4),(2,4),(4,4)))
policy = EpsilonGreedyPolicy(env, eps_0, eps_min, cutoff, initial_Q_values)

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
EvaluatePolicy(env,policy,number_of_runs=1,save_plots=True)
#env.fixed_initial_positions=None
