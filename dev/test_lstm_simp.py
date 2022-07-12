from modules.ppo.models_basic_lstm import GetEnv, Solve_LSTM, Solve, PPO_LSTM, PPO
from modules.dqn.dqn_utils import seed_everything

# Test worlds:
w1='MemoryTaskU1'               # Basic memory task, best score = 8.0
w2='MemoryTaskU1Long'           # Basic memory task with one extra step, best score = 7.0
w3='Manhattan3x3_WalkAround'    # Avoidance task, best score = 6.0

for seed in range(10):
    # Seeding
    seed_everything(seed)
    saveto='results/results_Phase3simp/test_lstm_simp1/SEED'+str(seed)+'/logs'

    # basic mem task, flat observation vector
    env = GetEnv(world_name=w1, num_frame_stack=1)
    model = PPO_LSTM(env, emb_dim=256)
    Solve_LSTM(env, model, logdir=saveto)

# # basic mem task, nfm/ei/reachable flat vector
# #nfm_func='NFM_ev_ec_t_dt_at_um_us'
# nfm_func='NFM_ev_ec_t_dt_at_ustack'
# env = CreateEnv('MemoryTaskU1', max_nodes=8, max_edges=22, nfm_func_name=nfm_func, var_targets=None, remove_world_pool=False, apply_wrappers=True, type_obs_wrap='Flat', obs_mask='freq', obs_rate=0.2)
# env_all_list = []
# env_all_list.append(env)

# super_env=SuperEnv(
#     [env],
#     hashint2env=None,
#     max_possible_num_nodes=8,
#     probs=[1])

# model = PPO_GNN_Single_LSTM(super_env, emb_dim=24)
# Solve_LSTM(env, model)