from environments import GraphWorld
from rl_policy import EpsilonGreedyPolicy
from rl_algorithms import q_learning, sarsa, expected_sarsa
from rl_plotting import PlotPerformanceCharts, PlotGridValues
import numpy as np
import simdata_utils as su

num_seeds   = 5
eps_0       = .1
eps_decay   = 0.
num_iter    = 500
gamma       = 1.
alpha_0     = .1
alpha_decay = 0.

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['Manhattan3']
conf['direction_north']=False

env = GraphWorld(conf)
env.sp.G = env.sp.G.to_directed()
policy = EpsilonGreedyPolicy(env, eps_0, eps_decay)

#env = SimpGraph()
#policy = EpsilonGreedyPolicy_graph(env, eps_0, eps_decay, 1)


metrics_episode_returns = {}
metrics_episode_lengths = {}
metrics_avgperstep = {}
Q_tables = {}

algos  = [q_learning,sarsa,expected_sarsa]
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

PlotPerformanceCharts(algos,performance_metrics)
PlotGridValues(algos,env,Q_tables)
