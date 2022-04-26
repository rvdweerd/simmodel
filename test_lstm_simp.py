from modules.ppo.models_basic_lstm import GetEnv, Solve_LSTM, Solve, PPO_LSTM, PPO
#Solve()

# basic mem task, flat observation vector
env = GetEnv()
model = PPO(env)
Solve(env,model)

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