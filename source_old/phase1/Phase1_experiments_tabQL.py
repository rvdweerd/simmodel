from modules.dqn.dqn_utils import seed_everything
from modules.rl.rl_policy import EpsilonGreedyPolicy
from modules.rl.rl_algorithms import q_learning_exhaustive
from modules.rl.rl_plotting import PlotPerformanceCharts, PlotGridValues
from modules.rl.rl_utils import EvaluatePolicy, GetFullCoverageSample
from modules.rl.rl_custom_worlds import CreateWorlds
import numpy as np
import pickle

num_seeds   = 1
eps_0       = 1.
eps_min     = 0.1
#cutoff      = 1#200
num_iter    = 1000
gamma       = .9
alpha_0     = .2
alpha_decay = 0.
initial_Q_values = 10.

run_world_names = [
    #'Manhattan3x3_PauseFreezeWorld',
    #'Manhattan3x3_PauseDynamicWorld',
    #'Manhattan5x5_DuplicateSetA',
    #'Manhattan5x5_DuplicateSetB',
    'Manhattan5x5_FixedEscapeInit',
    'Manhattan5x5_VariableEscapeInit',
    'MetroU3_e17tborder_FixedEscapeInit',
    #'MetroU3_e17t31_FixedEscapeInit', 
    #'MetroU3_e17t0_FixedEscapeInit', 
    #'MetroU3_e17tborder_VariableEscapeInit'
    'SparseManhattan5x5'
]
#worlds = CreateWorlds(run_world_names, make_reflexive=True, state_repr='et'      , state_enc='nodes')
#worlds = CreateWorlds(run_world_names, make_reflexive=True, state_repr='etUt'    , state_enc='nodes')
#worlds = CreateWorlds(run_world_names, make_reflexive=True, state_repr='ete0U0'   , state_enc='nodes')
worlds = CreateWorlds(run_world_names, make_reflexive=True, state_repr='etUte0U0', state_enc='nodes')
for w in worlds:
    w.capture_on_edges = False

for env, world_name in zip(worlds, run_world_names):
    policy = EpsilonGreedyPolicy(env, eps_0, eps_min, initial_Q_values)
    
    # # Learn the policy
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

    # Evaluate the learned policy
    
    # Load pre-trained Q table
    # logdir_load = 'results_Phase1/TabQL/'+world_name+'/'+env.state_representation
    # in_file = open(logdir_load + "/" + "Q_table.pkl", "rb")
    # in_dict = pickle.load(in_file)
    # in_file.close()
    # Qtable=in_dict['Q']
    # policy.Q=Qtable

    policy.epsilon=0.
    edgeblock=env.capture_on_edges
    logdir = 'results/testing/TabQL/'+world_name+'/eblock'+str(edgeblock)+'/'+env.state_representation
    _, returns, captures, solves = EvaluatePolicy(env,policy,env.world_pool,print_runs=False, save_plots=False, logdir=logdir, has_Q_table=True)
    plotlist = GetFullCoverageSample(returns, env.world_pool, bins=10, n=10)
    EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir=logdir, has_Q_table=True)
    saveinfo = {"Q":policy.Q}
    a_file = open(logdir + "/" + "Q_table.pkl", "wb")
    pickle.dump(saveinfo, a_file)
    a_file.close()