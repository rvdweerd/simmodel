from environments import GraphWorld
from rl_policy import EpsilonGreedyPolicy, RandomPolicy, ShortestPathPolicy
from rl_algorithms import q_learning, sarsa, expected_sarsa, q_learning_exhaustive
from rl_plotting import PlotPerformanceCharts, PlotGridValues, PlotNodeValues
from rl_utils import EvaluatePolicy, CreateDuplicatesTrainsets
from rl_custom_worlds import GetCustomWorld
import numpy as np
import simdata_utils as su
import random 

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
#conf=configs['Manhattan5']
#conf=configs['Manhattan11']
#conf=configs['MetroGraphU3L8_node1']
conf=configs['MetroGraphU4L8_node1']
conf['direction_north']=False
conf['loadAllStartingPositions']=False
conf['make_reflexive']=False
env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation='etUte0U0')

#env=GetCustomWorld('Manhattan3x3_PauseFreezeWorld')
#env._encode=env._encode_nodes
#env.state_representation = 'et'

# CODE TO CREATE DUPLICATE SETS
#init_pos_trainset_indices0, init_pos_trainset_indices1 = CreateDuplicatesTrainsets(env, min_y_coord=env.sp.N-1, min_num_same_positions=env.sp.U, min_num_worlds=4, print_selection=False)
#env.world_pool = init_pos_trainset_indices1 # limit the training set to the selected entries
# Select full world pool
#env.world_pool = env.all_worlds

policy_random   = RandomPolicy(env)
policy_shortest = ShortestPathPolicy(env, weights = 'equal')
policy_mindeg   = ShortestPathPolicy(env, weights = 'min_indegree')

env.sp.target_nodes=[31]
#env.world_pool = env.all_worlds

#env.max_timesteps=env.sp.L
pr=False
sp=False
EvaluatePolicy(env, policy_random  , env.world_pool*20, print_runs=pr, save_plots=sp)
#EvaluatePolicy(env, policy_random  , [142,179], print_runs=True, save_plots=True)

EvaluatePolicy(env, policy_shortest, env.world_pool, print_runs=pr, save_plots=sp)
#EvaluatePolicy(env, policy_shortest, [0,1,2,3,4,5,6,7,8,9], print_runs=True, save_plots=True)

EvaluatePolicy(env, policy_mindeg  , env.world_pool, print_runs=pr, save_plots=sp)
#EvaluatePolicy(env, policy_mindeg  , [0], print_runs=True, save_plots=True)
