import simdata_utils as su
from environments import GraphWorld
import random
import simdata_utils as su
from rl_utils import EvaluatePolicy
from rl_policy import LeftUpPolicy, RandomPolicy

random.seed(42)

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
#conf=configs['Manhattan3']
#conf=configs['Manhattan5']
conf=configs['Manhattan11']
#conf=configs['CircGraph']
#conf=configs['TKGraph']
conf['direction_north']=False

env=GraphWorld(conf, optimization_method='static', fixed_initial_positions=None)
a=10
for k,v in env.register['coords'].items():
    if k[0][1]==0:
        if k[1][1]>=a and k[2][1]>=a and k[3][1]>=a:
            print(k,v)
