from environments import GraphWorld
from rl_policy import EpsilonGreedyPolicy, RandomPolicy, ShortestPathPolicy
from rl_algorithms import q_learning, sarsa, expected_sarsa, q_learning_exhaustive
from rl_plotting import PlotPerformanceCharts, PlotGridValues, PlotNodeValues
from rl_utils import EvaluatePolicy, CreateDuplicatesTrainsets
import numpy as np
import simdata_utils as su
import random 

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
#conf=configs['Manhattan5']
conf=configs['MetroGraphU3L8_node1']
#conf=configs['Manhattan11']
conf['direction_north']=False
conf['loadAllStartingPositions']=False
conf['make_reflexive']=True
env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation='etUte0U0')

# CODE TO CREATE BACKTRACK EXAMPLE
# env.register['coords']={((1,0),(0,2),(1,2),(2,2)):0}
# env.register['labels']={(1,6,7,8):0}
# env.databank['coords']=[{'start_escape_route':(1,0), 'start_units':[(0,2),(1,2),(2,2)], 'paths':[[(0,2),(0,1),(0,0),(0,1),(0,0)],[(1,2),(1,2),(1,2),(0,2),(0,1)],[(2,2),(2,1)]]}]
# env.databank['labels']=[{'start_escape_route':1, 'start_units':[6,7,8], 'paths':[[6,3,0,3,0],[7,7,7,6,3],[8,5]]}]
# env.iratios=[1.]
# env.all_worlds=[0]
# env.world_pool=[0]

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
EvaluatePolicy(env, policy_random  , env.world_pool, print_runs=pr, save_plots=sp)
#EvaluatePolicy(env, policy_random  , [142,179], print_runs=True, save_plots=True)

EvaluatePolicy(env, policy_shortest, env.world_pool, print_runs=pr, save_plots=sp)
#EvaluatePolicy(env, policy_shortest, [142,179], print_runs=True, save_plots=True)

EvaluatePolicy(env, policy_mindeg  , env.world_pool, print_runs=pr, save_plots=sp)
#EvaluatePolicy(env, policy_mindeg  , [142,179], print_runs=True, save_plots=True)
