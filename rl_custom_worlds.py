import simdata_utils as su
from environments import GraphWorld

def GetCustomWorld(world_name):
    if world_name == 'Manhattan3x3_PauseFreezeWorld':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan3']
        conf['direction_north']=False
        conf['make_reflexive']=True
        conf['U']=3
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation='et', state_encoding='tensor')
        env.register['coords']={((1,0),(0,2),(1,2),(2,2)):0}
        env.register['labels']={(1,6,7,8):0}
        env.databank['coords']=[{'start_escape_route':(1,0), 'start_units':[(0,2),(1,2),(2,2)], 'paths':[[(0,2),(0,1)],[(1,2),(1,2),(1,2),(0,2)],[(2,2),(2,1)]]}]
        env.databank['labels']=[{'start_escape_route':1, 'start_units':[6,7,8], 'paths':[[6,3],[7,7,7,6],[8,5]]}]
        env.iratios=[1.]
        env.all_worlds=[0]
        env.world_pool=[0]
        return env
    if world_name == 'Manhattan3x3_PauseDynamicWorld':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan3']
        conf['direction_north']=False
        conf['make_reflexive']=True
        conf['U']=3
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation='et', state_encoding='tensor')
        env.register['coords']={((1,0),(0,2),(1,2),(2,2)):0}
        env.register['labels']={(1,6,7,8):0}
        env.databank['coords']=[{'start_escape_route':(1,0), 'start_units':[(0,2),(1,2),(2,2)], 'paths':[[(0,2),(0,1),(0,0)],[(1,2),(1,1),(0,1)],[(2,2),(2,1)]]}]
        env.databank['labels']=[{'start_escape_route':1, 'start_units':[6,7,8], 'paths':[[6,3,0],[7,4,3],[8,5]]}]
        env.iratios=[1.]
        env.all_worlds=[0]
        env.world_pool=[0]
        return env
    if world_name == 'Manhattan3x3_TestWorld':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan3']
        conf['direction_north']=False
        conf['make_reflexive']=True
        conf['U']=3
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation='et', state_encoding='tensor')
        env.register['coords']={((1,0),(0,2),(1,2),(2,2)):0}
        env.register['labels']={(1,6,7,8):0}
        env.databank['coords']=[{'start_escape_route':(1,0), 'start_units':[(0,2),(1,2),(2,2)], 'paths':[[(0,2)],[(1,2),(0,2)],[(2,2)]]}]
        env.databank['labels']=[{'start_escape_route':1, 'start_units':[6,7,8], 'paths':[[6],[7,6],[8]]}]
        env.iratios=[1.]
        env.all_worlds=[0]
        env.world_pool=[0]
        return env


