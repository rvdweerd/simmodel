import matplotlib.pyplot as plt
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.sim.simdata_utils import SimulateInteractiveMode
from modules.ppo.ppo_wrappers import PPO_ObsBasicDictWrapper, PPO_ObsBasicDictWrapperCRE
import matplotlib.pyplot as plt
import modules.gnn.nfm_gen
import pickle

# world_list=[
#     'NWB_test_FixedEscapeInit',
#     'Manhattan5x5_FixedEscapeInit',
#     'Manhattan3x3_PauseFreezeWorld',
#     'MetroU3_e17tborder_FixedEscapeInit',
#     'Manhattan11x11',
#     'NWB_ROT_FixedEscapeInit',
#     'NWB_UTR_FixedEscapeInit',
# ]
# target_node_info={}
# target_node_info['world_list']=world_list
# graph_type_list=[]
# for w in world_list:
#     env = GetCustomWorld(w, make_reflexive=True, state_repr='et', state_enc='tensors')
#     gtype=env.sp.graph_type+'_N='+str(env.sp.N)
#     graph_type_list.append(gtype)
#     info={
#         'N':env.sp.N,
#         'U':env.sp.U,
#         'L':env.sp.L,
#         'R':env.sp.R,
#         'target_nodes_labels':env.sp.target_nodes,
#         'target_nodes_coord':[env.sp.labels2coord[l] for l in env.sp.target_nodes],
#     }
#     target_node_info[gtype]=info
# out_file = open("datasets/target_node_info.pkl", "wb")
# pickle.dump(target_node_info, out_file)
# out_file.close()

#world_name='CircGraph'
#world_name='TKGraph'
#world_name='BifurGraphTask1'
world_name='M3test1'
#world_name='MemoryTaskU1'
#world_name='MetroU3_e17tborder_FixedEscapeInit'
#world_name='Manhattan11x11_FixedEscapeInit'
#world_name='MetroU1_e17tborder_VariableEscapeInit'
#world_name='NWB_test_FixedEscapeInit'
#world_name='NWB_ROT_FixedEscapeInit'
#world_name='NWB_ROT_FixedEscapeInit2'
#world_name='NWB_UTR_FixedEscapeInit'
#world_name='NWB_UTR_FixedEscapeInit2'
#world_name='Manhattan3x3_WalkAround'
#world_name='Manhattan5x5_FixedEscapeInit'
#world_name='SparseManhattan5x5'
N=9#27#975#33#25#3975
E=9#52#3000#105#4000
obs_mask='None'
obs_rate=1.0
state_repr='etUte0U0'
state_enc='nfm'
nfm_func=modules.gnn.nfm_gen.nfm_funcs['NFM_ev_ec_t_dt_at_um_us']
#nfm_func=modules.gnn.nfm_gen.nfm_funcs['NFM_ev_ec_t_dt_at_ustack']
env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
env.redefine_nfm(nfm_func)
#env.redefine_goal_nodes([])
#env=PPO_ObsFlatWrapper(env, max_possible_num_nodes=N, max_possible_num_edges=E, obs_mask=obs_mask, obs_rate=obs_rate)
env=PPO_ObsBasicDictWrapper(env, obs_mask=obs_mask, obs_rate=obs_rate)

for epi in range(0):
    done=False
    o=env.reset()
    while not done:
        print(o[:N*nfm_func.F].reshape(N,nfm_func.F),'\n')
        print(env.nfm,'\n')
        env.render(fname='test_0step',t_suffix=False)
        env.render_eupaths(fname='test_1step', t_suffix=False, last_step_only=True)
        o,r,done,info = env.step(0)

while True:
    print('Shortest path', env.sp.spath_to_target,'length:', env.sp.spath_length,'hops')   
    SimulateInteractiveMode(env, filesave_with_time_suffix=False, entry=None)

