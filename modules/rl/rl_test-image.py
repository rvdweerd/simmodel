#import networkx as nx
from modules.rl.environments import GraphWorld
#from rl_policy import EpsilonGreedyPolicy
#from rl_algorithms import q_learning, sarsa, expected_sarsa, q_learning_exhaustive
#from rl_plotting import PlotPerformanceCharts, PlotGridValues, PlotNodeValues
#from rl_utils import EvaluatePolicy, CreateDuplicatesTrainsets
#import numpy as np
#from modules.sim.sim_graphs import MetroGraph
import modules.sim.simdata_utils as su
#import random 

#G, labels, pos = MetroGraph_ext()
configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['MetroGraphU4']
conf['direction_north']=False
env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation='et')
env.render(fname='test-image')