import simdata_utils as su
from environments import GraphWorld

def GetCustomWorld(world_name, make_reflexive=True, state_repr='et', state_enc='nodes'):
    if world_name == 'Manhattan3x3_PauseFreezeWorld':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan3']
        conf['direction_north']=False
        conf['make_reflexive']=make_reflexive
        conf['U']=3
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
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
        conf['make_reflexive']=make_reflexive
        conf['U']=3
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
        env.register['coords']={((1,0),(0,2),(1,2),(2,2)):0}
        env.register['labels']={(1,6,7,8):0}
        env.databank['coords']=[{'start_escape_route':(1,0), 'start_units':[(0,2),(1,2),(2,2)], 'paths':[[(0,2),(0,1),(0,0)],[(1,2),(1,1),(0,1)],[(2,2),(2,1)]]}]
        env.databank['labels']=[{'start_escape_route':1, 'start_units':[6,7,8], 'paths':[[6,3,0],[7,4,3],[8,5]]}]
        env.iratios=[1.]
        env.all_worlds=[0]
        env.world_pool=[0]
        return env
    if world_name == 'Manhattan5x5_FixedEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan5']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env
    if world_name == 'Manhattan5x5_VariableEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan5']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=True
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env
    if world_name == 'MetroU3_e17tborder_FixedEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MetroGraphU3']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        assert conf['T'] ==  20
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env
    if world_name == 'MetroU3_e17t31_FixedEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MetroGraphU3']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        conf['make_reflexive']=make_reflexive
        assert  conf['T'] == 20
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        env.sp.target_nodes=[31]
        return env
    if world_name == 'MetroU3_e17t0_FixedEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MetroGraphU3']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        conf['make_reflexive']=make_reflexive
        assert  conf['T'] == 20
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        env.sp.target_nodes=[0]
        return env
    if world_name == 'MetroU3_e1t31_FixedEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MetroGraphU3L8_node1']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        conf['make_reflexive']=make_reflexive
        assert conf['T'] == 25
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        env.sp.target_nodes=[31]
        return env

def CreateWorlds(run_world_names, make_reflexive=True, state_repr='et', state_enc='nodes'):
    worlds=[]
    for world_name in run_world_names:
        worlds.append(GetCustomWorld(world_name, make_reflexive, state_repr, state_enc))
    return worlds
