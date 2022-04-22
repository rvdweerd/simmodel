import argparse
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from modules.dqn.dqn_utils import seed_everything
from modules.sim.graph_factory import GetWorldSet
from modules.ppo.ppo_custom_SIMP import *
from modules.ppo.helpfuncs import CreateEnv, evaluate_lstm_ppo
from modules.rl.rl_utils import GetFullCoverageSample
from modules.rl.rl_policy import LSTM_GNN_PPO_Policy, LSTM_GNN_PPO_EMB_Policy
from modules.rl.rl_utils import EvaluatePolicy
from modules.sim.simdata_utils import SimulateAutomaticMode_PPO, SimulateInteractiveMode_PPO
from modules.ppo.ppo_wrappers import PPO_ActWrapper, PPO_ObsFlatWrapper
torch.set_num_threads(1) # Max #threads for torch to avoid inefficient util of cpu cores.

def main(args):
    config, hp, tp = GetConfigs(args)   
    
    if config['train']:
        train_env, _ = make_custom(config, num_envs=hp.parallel_rollouts, asynchronous=tp['asynchronous_environment'])
        o=train_env.reset()
        hp.max_possible_nodes = 8
        hp.max_possible_edges = 22

        if config['demoruns']:
            while True:
                a = SimulateInteractiveMode_PPO(train_env.envs[0], filesave_with_time_suffix=False)
                if a == 'Q': break

        
        WriteTrainParamsToFile(config,hp,tp)
        for seed in config['seedrange']:
            seed_everything(seed)
            train_env.seed(seed)
            logdir_=config['logdir']+'/SEED'+str(seed)
            tp['writer'] = SummaryWriter(log_dir=f"{logdir_}/logs")
            tp["base_checkpoint_path"]=f"{logdir_}/checkpoints/"
            tp["seed_path"]=logdir_
            #tp["workspace_path"]=logdir_

            ppo_model, ppo_optimizer, iteration, stop_conditions = start_or_resume_from_checkpoint(train_env, config, hp, tp)
            if seed == config['seed0']: WriteModelParamsToFile(config, ppo_model)
            score = train_model_EMB_SIMP(train_env, ppo_model, ppo_optimizer, iteration, stop_conditions, hp, tp)



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
    parser.add_argument('--eval_deter', type=lambda s: s.lower() in ['true', 't', 'yes', '1'],default=True)
    args=parser.parse_args()
    main(args)
