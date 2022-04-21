import argparse
#from aiohttp import HttpVersion10
import matplotlib.pyplot as plt
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.sim.simdata_utils import SimulateInteractiveMode
from modules.ppo.ppo_wrappers import PPO_ObsFlatWrapper, PPO_ActWrapper
import matplotlib.pyplot as plt
import modules.gnn.nfm_gen
from modules.ppo.ppo_custom import *
from modules.rl.rl_policy import LSTM_GNN_PPO_Policy
from modules.sim.simdata_utils import SimulateAutomaticMode_PPO, SimulateAutomaticMode_PPO_LSTM, SimulateInteractiveMode_PPO

def main(args):
    config, hp, tp = GetConfigs(args)   

    world_name='Manhattan5x5_FixedEscapeInit'
    N=25#33
    E=105#150
    obs_mask='prob_per_u_test'
    obs_rate=0.5
    state_repr='etUte0U0'
    state_enc='nfm'
    #assert config['lstm_type']
    nfm_func=modules.gnn.nfm_gen.nfm_funcs['NFM_ev_ec_t_dt_at_um_us']
    #nfm_func=modules.gnn.nfm_gen.nfm_funcs['NFM_ev_ec_t_dt_at_ustack']
    env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    env.redefine_nfm(nfm_func)
    env = PPO_ObsFlatWrapper(env, max_possible_num_nodes=N, max_possible_num_edges=E, obs_mask=obs_mask, obs_rate=obs_rate)
    env = PPO_ActWrapper(env)

    train_env, env_all_list = make_custom(config, num_envs=1, asynchronous=tp['asynchronous_environment'])

    env_ = env
    hp.max_possible_nodes = env_.max_possible_num_nodes
    hp.max_possible_edges = env_.max_possible_num_edges
    seed = config['seed0']
    logdir_=config['logdir']+'/SEED'+str(seed)
    tp["base_checkpoint_path"]=f"{logdir_}/checkpoints/"
    assert  os.path.exists(tp['base_checkpoint_path'])
    tp['eval_deterministic']=False

    ppo_model_lstm, _,_,_ = start_or_resume_from_checkpoint(train_env, config, hp, tp)
    ppo_policy_lstm = LSTM_GNN_PPO_Policy(None, ppo_model_lstm, deterministic=tp['eval_deterministic'])
    
    hp2=copy.deepcopy(hp)
    config2=copy.deepcopy(config)
    tp2=copy.deepcopy(tp)
    config2['lstm_type']='None'
    hp2.lstm_on = False
    hp2.critic='v'
    tp2['base_checkpoint_path'] = './results/results_Phase3/ppo/M5x5Fixed/gat2-v/emb24_itT5/lstm_None/NFM_ev_ec_t_dt_at_um_us/omask_prob_per_u0.75/bsize64/SEED15/checkpoints/'
    tp2['eval_deterministic'] = False
    
    ppo_model_no_lstm, _,_,_ = start_or_resume_from_checkpoint(train_env, config2, hp2, tp2)
    ppo_policy_no_lstm = LSTM_GNN_PPO_Policy(None, ppo_model_no_lstm, deterministic=tp2['eval_deterministic'])
    
    while True:
        entries=[1680]#None#[831]#[5012,218,3903]
        #demo_env = random.choice(evalenv)
        a = SimulateAutomaticMode_PPO_LSTM(env_, ppo_policy_lstm, ppo_policy_no_lstm, t_suffix=False, entries=entries)
        if a == 'Q': break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--train_on', default='None', type=str)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--recurrent_seq_len', default=2, type=int)
    parser.add_argument('--parallel_rollouts', default=1, type=int)
    parser.add_argument('--rollout_steps', default=100, type=int)
    parser.add_argument('--patience', default=500, type=int)
    parser.add_argument('--obs_mask', default='None', type=str, help='U obervation masking type', choices=['None','freq','prob','prob_per_u'])
    parser.add_argument('--obs_rate', default=1.0, type=float)
    parser.add_argument('--emb_dim', default=64, type=int)
    parser.add_argument('--lstm_type', default='None', type=str, choices=['None','EMB','FE','Dual','DualCC'])
    parser.add_argument('--lstm_hdim', default=64, type=int)
    parser.add_argument('--lstm_layers', default=1, type=int)
    #parser.add_argument('--lstm_dropout', default=0.0, type=float)
    parser.add_argument('--emb_iterT', default=2, type=int)
    parser.add_argument('--nfm_func', default='NFM_ev_ec_t', type=str)
    parser.add_argument('--qnet', default='gat2', type=str)
    parser.add_argument('--critic', default='q', type=str, choices=['q','v']) # q=v value route, v=single value route
    parser.add_argument('--train', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--eval', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])   
    parser.add_argument('--test', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])       
    parser.add_argument('--num_seeds', default=5, type=int)
    parser.add_argument('--seed0', default=10, type=int)
    parser.add_argument('--demoruns', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--eval_deter', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    args=parser.parse_args()
    main(args)
