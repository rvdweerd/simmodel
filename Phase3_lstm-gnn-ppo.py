import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from modules.dqn.dqn_utils import seed_everything
from modules.ppo.ppo_custom import *
from modules.ppo.helpfuncs import CreateEnv
from modules.rl.rl_utils import GetFullCoverageSample
from modules.rl.rl_policy import LSTM_GNN_PPO_Policy
from modules.rl.rl_utils import EvaluatePolicy
from modules.sim.simdata_utils import SimulateAutomaticMode_PPO
torch.set_num_threads(4) # Max #threads for torch to avoid inefficient util of cpu cores.


def TestSavedModel(config, hp, tp):
    #env=GetCustomWorld(WORLD_NAME, make_reflexive=MAKE_REFLEXIVE, state_repr=STATE_REPR, state_enc='tensors')
    world_name='Manhattan3x3_WalkAround'
    env = CreateEnv(world_name, max_nodes=9, max_edges = 33, nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=None, apply_wrappers=True)
    def envf():
        return env
    env_ = SyncVectorEnv([envf])
    #env_ = make_custom(world_name, num_envs=1, asynchronous=False)   
    ppo_model, ppo_optimizer, max_checkpoint_iteration, stop_conditions = start_or_resume_from_checkpoint(env_, config, hp, tp)
    
    #env=env_.envs[0]
    policy = LSTM_GNN_PPO_Policy(env, ppo_model, deterministic=tp['eval_deterministic'])
    while True:
        entries=None#[5012,218,3903]
        #demo_env = random.choice(evalenv)
        a = SimulateAutomaticMode_PPO(env, policy, t_suffix=False, entries=entries)
        if a == 'Q': break
    
    #lengths, returns, captures, solves = EvaluatePolicy(env, policy  , env.world_pool*SAMPLE_MULTIPLIER, print_runs=False, save_plots=False, logdir=exp_rootdir)    
    #plotlist = GetFullCoverageSample(returns, env.world_pool*SAMPLE_MULTIPLIER, bins=10, n=10)
    #EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir=exp_rootdir)

def main(args):
    config, hp, tp = GetConfigs(args)   
    
    if config['train']:
        train_env = make_custom(config, num_envs=hp.parallel_rollouts, asynchronous=tp['asynchronous_environment'])
        hp.max_possible_nodes = train_env.envs[0].env.max_possible_num_nodes
        hp.max_possible_edges = train_env.envs[0].env.max_possible_num_edges
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
            score = train_model(train_env, ppo_model, ppo_optimizer, iteration, stop_conditions, hp, tp)

    if config['eval']:
        train_env = make_custom(config, num_envs=1, asynchronous=tp['asynchronous_environment'])
        hp.max_possible_nodes = train_env.envs[0].env.max_possible_num_nodes
        hp.max_possible_edges = train_env.envs[0].env.max_possible_num_edges
        seed = config['seed0']
        logdir_=config['logdir']+'/SEED'+str(seed)
        tp["base_checkpoint_path"]=f"{logdir_}/checkpoints/"
        ppo_model, ppo_optimizer, max_checkpoint_iteration, stop_conditions = start_or_resume_from_checkpoint(train_env, config, hp, tp)
        env = train_env.envs[0]
        policy = LSTM_GNN_PPO_Policy(env, ppo_model, deterministic=tp['eval_deterministic'])
        while True:
            entries=None#[5012,218,3903]
            #demo_env = random.choice(evalenv)
            a = SimulateAutomaticMode_PPO(env, policy, t_suffix=False, entries=entries)
            if a == 'Q': break

    if config['test']:
        seed = config['seed0']
        logdir_=config['logdir']+'/SEED'+str(seed)
        tp["base_checkpoint_path"]=f"{logdir_}/checkpoints/"
        #tp["workspace_path"]=logdir_
        TestSavedModel(config, hp, tp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--train_on', default='None', type=str)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--recurrent_seq_len', default=2, type=int)
    parser.add_argument('--parallel_rollouts', default=4, type=int)
    parser.add_argument('--rollout_steps', default=201, type=int)
    parser.add_argument('--patience', default=500, type=int)
    parser.add_argument('--mask_type', default='None', type=str, help='U obervation masking type', choices=['None','freq','prob'])
    parser.add_argument('--mask_rate', default=1.0, type=float)
    parser.add_argument('--emb_dim', default=64, type=int)
    parser.add_argument('--lstm_type', default='shared-noncat', type=str, choices=['None','shared-concat','shared-noncat','separate-noncat'])
    parser.add_argument('--lstm_hdim', default=64, type=int)
    parser.add_argument('--lstm_layers', default=1, type=int)
    #parser.add_argument('--lstm_dropout', default=0.0, type=float)
    parser.add_argument('--emb_iterT', default=2, type=int)
    parser.add_argument('--nfm_func', default='NFM_ev_ec_t', type=str)
    parser.add_argument('--qnet', default='gat2', type=str)
    parser.add_argument('--train', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--eval', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])   
    parser.add_argument('--test', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])       
    parser.add_argument('--num_seeds', default=5, type=int)
    parser.add_argument('--seed0', default=10, type=int)
    parser.add_argument('--demoruns', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    args=parser.parse_args()
    main(args)
