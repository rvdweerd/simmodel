from environments import GraphWorld
from rl_policy import EpsilonGreedyPolicy, MinIndegreePolicy, RandomPolicy
from rl_algorithms import q_learning, sarsa, expected_sarsa, q_learning_exhaustive
from rl_plotting import PlotPerformanceCharts, PlotGridValues, PlotNodeValues
from rl_utils import EvaluatePolicy, CreateDuplicatesTrainsets
import numpy as np
import simdata_utils as su
import random 

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['Manhattan5']
conf['direction_north']=False

env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation='etUte0U0')
policy_mindeg = MinIndegreePolicy(env)
policy_random = RandomPolicy(env)
#env.world_pool = env.all_worlds

EvaluatePolicy(env,policy_mindeg,[100],print_runs=False, save_plots=True)
#env.max_timesteps=env.sp.L
EvaluatePolicy(env,policy_random,env.world_pool*1,print_runs=False, save_plots=False)
