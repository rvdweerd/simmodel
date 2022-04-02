import matplotlib.pyplot as plt
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.sim.simdata_utils import SimulateInteractiveMode
from modules.ppo.ppo_wrappers import PPO_ObsFlatWrapper
import matplotlib.pyplot as plt
import modules.gnn.nfm_gen

#world_name='Manhattan3x3_PauseFreezeWorld'
#world_name='Manhattan3x3_PauseDynamicWorld'
#world_name='Manhattan3x3_WalkAround'
#world_name='Manhattan5x5_FixedEscapeInit'
world_name='MetroU3_e17tborder_FixedEscapeInit'
#world_name='MetroU3_e17t0_FixedEscapeInit'
#world_name='MetroU3_e1t31_FixedEscapeInit'
#world_name='SparseManhattan5x5'
#world_name = 'NWB_test_FixedEscapeInit'
#world_name='NWB_ROT_FixedEscapeInit'
N=33
E=150
obs_mask='prob_per_u'
obs_rate=0.0
state_repr='etUte0U0'
state_enc='nfm'
#nfm_func=modules.gnn.nfm_gen.nfm_funcs['NFM_ev_ec_t_dt_at_um_us']
nfm_func=modules.gnn.nfm_gen.nfm_funcs['NFM_ev_ec_t_dt_at_ustack']
env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
env.redefine_nfm(nfm_func)
env=PPO_ObsFlatWrapper(env, max_possible_num_nodes=N, max_possible_num_edges=E, obs_mask=obs_mask, obs_rate=obs_rate)

for epi in range(5):
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
    SimulateInteractiveMode(env, filesave_with_time_suffix=False)

