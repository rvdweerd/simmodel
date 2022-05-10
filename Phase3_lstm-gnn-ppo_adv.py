import argparse
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from modules.dqn.dqn_utils import seed_everything
from modules.sim.graph_factory import GetWorldSet
from modules.ppo.ppo_custom import *
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
        hp.max_possible_nodes = int(o[0,-4])
        hp.max_possible_edges = int(o[0,-2])
        assert int(o[0,-4]) == train_env.envs[0].env.max_possible_num_nodes
        assert int(o[0,-2]) == train_env.envs[0].env.max_possible_num_edges

        if config['demoruns']:
            while True:
                a = SimulateInteractiveMode_PPO(train_env.envs[0], filesave_with_time_suffix=False)
                if a == 'Q': break

        
        WriteTrainParamsToFile(config,hp,tp)
        trainfuncmap={'FE':train_model_FE,'EMB':train_model_EMB,'None':train_model,'Dual':train_model,'DualCC':train_model}
        trainfunc = trainfuncmap[config['lstm_type']]
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
            score = trainfunc(train_env, ppo_model, ppo_optimizer, iteration, stop_conditions, hp, tp)

    if config['eval']:
        train_env, env_all_list = make_custom(config, num_envs=1, asynchronous=tp['asynchronous_environment'])
        env_ = train_env.envs[0]
        hp.max_possible_nodes = train_env.envs[0].env.max_possible_num_nodes
        hp.max_possible_edges = train_env.envs[0].env.max_possible_num_edges
        seed = config['seed0']
        logdir_=config['logdir']+'/SEED'+str(seed)
        tp["base_checkpoint_path"]=f"{logdir_}/checkpoints/"
        assert  os.path.exists(tp['base_checkpoint_path'])
        
        if config['demoruns']:
            ppo_model, ppo_optimizer, max_checkpoint_iteration, stop_conditions = start_or_resume_from_checkpoint(train_env, config, hp, tp)
            if config['lstm_type'] == 'EMB':
                ppo_policy = LSTM_GNN_PPO_EMB_Policy(None, ppo_model, deterministic=tp['eval_deterministic'])
            else:
                ppo_policy = LSTM_GNN_PPO_Policy(None, ppo_model, deterministic=tp['eval_deterministic'])
            while True:
                entries=None#[5012,218,3903]
                #demo_env = random.choice(evalenv)
                a = SimulateAutomaticMode_PPO(env_, ppo_policy, t_suffix=False, entries=entries)
                if a == 'Q': break

        evalResults={}
        evalName='trainset'+'_evaldet'+str(tp['eval_deterministic'])[0]
        evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
        for seed in config['seedrange']:
            logdir_=config['logdir']+'/SEED'+str(seed)
            tp["base_checkpoint_path"]=f"{logdir_}/checkpoints/"
            try:
                assert os.path.exists(tp['base_checkpoint_path'])
            except:
                continue
            ppo_model, ppo_optimizer, max_checkpoint_iteration, stop_conditions = start_or_resume_from_checkpoint(train_env, config, hp, tp)
            if config['lstm_type'] == 'EMB':
                ppo_policy = LSTM_GNN_PPO_EMB_Policy(None, ppo_model, deterministic=tp['eval_deterministic'])
            else:
                ppo_policy = LSTM_GNN_PPO_Policy(None, ppo_model, deterministic=tp['eval_deterministic'])

            multiplier=1
            if not tp['eval_deterministic']:
                k=sum([len(k.world_pool) for k in env_all_list])
                multiplier = max(1,20000//k)
            result = evaluate_lstm_ppo(logdir=logdir_, config=config, env = env_all_list, ppo_policy=ppo_policy, eval_subdir=evalName, max_num_nodes=hp.max_possible_nodes, multiplier=multiplier)
            num_unique_graphs, num_graph_instances, avg_return, success_rate = result
            evalResults[evalName]['num_graphs.........'].append(num_unique_graphs)
            evalResults[evalName]['num_graph_instances'].append(num_graph_instances)
            evalResults[evalName]['avg_return.........'].append(avg_return)
            evalResults[evalName]['success_rate.......'].append(success_rate)

        for ename, results in evalResults.items():
            OF = open(config['logdir']+'/Eval_det'+str(tp['eval_deterministic'])[0]+'_Results_over_seeds_'+ename+'.txt', 'w')
            def printing(text):
                print(text)
                OF.write(text + "\n")
            np.set_printoptions(formatter={'float':"{0:0.3f}".format})
            printing('Results over seeds for evaluation on '+ename+'\n')
            for category,values in results.items():
                printing(category)
                printing('  avg over seeds: '+str(np.mean(values)))
                printing('  std over seeds: '+str(np.std(values)))
                printing('  per seed: '+str(np.array(values))+'\n')

    if config['test']:
        evalResults={}
        world_dict={ # [max_nodes,max_edges]
            #'Manhattan5x5_DuplicateSetB':[25,300],
            #'Manhattan3x3_WalkAround':[9,300],
            #'MetroU3_e1t31_FixedEscapeInit':[33, 300],
            #'full_solvable_3x3subs':[9,33],
            # 'MemoryTaskU1':[8,16],
            'Manhattan5x5_FixedEscapeInit':[25,105],
            'Manhattan5x5_VariableEscapeInit':[25,105],
            'MetroU3_e17tborder_FixedEscapeInit':[33,300],
            'MetroU3_e17tborder_VariableEscapeInit':[33,300],
            'NWB_ROT_FixedEscapeInit':[2602,7300],
            'NWB_ROT_VariableEscapeInit':[2602,7300],
            'NWB_test_FixedEscapeInit':[975,4000],
            'NWB_test_VariableEscapeInit':[975,4000],
            'NWB_UTR_FixedEscapeInit':[1182,4000],
            'NWB_UTR_VariableEscapeInit':[1182,4000],
            # 'SparseManhattan5x5':[25,105],
            }
        obs_evalmasks = ['None']#'prob_per_u_test','prob_per_u_test','prob_per_u_test','prob_per_u_test','prob_per_u_test'] # ['None']['prob_per_u']
        obs_evalrates = [1.0]#0.9,.8,.7,.6,.5]    # [1.][0.8]
        for obs_mask, obs_rate in zip(obs_evalmasks, obs_evalrates):
            for world_name in world_dict.keys():
                evalName=world_name+'_obs'+obs_mask+'_evaldet'+str(tp['eval_deterministic'])[0]
                if obs_mask != 'None': evalName += str(obs_rate)
                if world_name == "full_solvable_3x3subs":
                    Etest=[0,1,2,3,4,5,6,7,8,9,10]
                    Utest=[1,2,3]
                    evalenv, _, _, _  = GetWorldSet('etUte0U0', 'nfm', U=Utest, E=Etest, edge_blocking=config['edge_blocking'], solve_select=config['solve_select'], reject_duplicates=False, nfm_func=modules.gnn.nfm_gen.nfm_funcs[config['nfm_func']], apply_wrappers=False, maxnodes=world_dict[world_name][0], maxedges=world_dict[world_name][1])
                    for i in range(len(evalenv)):
                        evalenv[i]=PPO_ObsFlatWrapper(evalenv[i], max_possible_num_nodes=world_dict[world_name][0], max_possible_num_edges=world_dict[world_name][1], obs_mask=obs_mask, obs_rate=obs_rate)
                        evalenv[i]=PPO_ActWrapper(evalenv[i])
                    env=evalenv[0]
                else:
                    env = CreateEnv(world_name, max_nodes=world_dict[world_name][0], max_edges = world_dict[world_name][1], nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=None, apply_wrappers=True, type_obs_wrap=config['type_obs_wrap'], obs_mask=obs_mask, obs_rate=obs_rate)
                    evalenv=[env]
                hp.max_possible_nodes = world_dict[world_name][0]#env.max_possible_num_nodes
                hp.max_possible_edges = world_dict[world_name][1]#env.max_possible_num_edges
                def envf():
                    return env
                env_ = SyncVectorEnv([envf])

                evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
                for seed in config['seedrange']:
                    logdir_=config['logdir']+'/SEED'+str(seed)
                    tp["base_checkpoint_path"]=f"{logdir_}/checkpoints/"
                    try:
                        assert os.path.exists(tp['base_checkpoint_path'])
                    except:
                        continue
                    ppo_model, ppo_optimizer, max_checkpoint_iteration, stop_conditions = start_or_resume_from_checkpoint(env_, config, hp, tp)
                    if config['lstm_type'] == 'EMB':
                        policy = LSTM_GNN_PPO_EMB_Policy(None, ppo_model, deterministic=tp['eval_deterministic'])
                    else:                   
                        ppo_policy = LSTM_GNN_PPO_Policy(env, ppo_model, deterministic=tp['eval_deterministic'])

                    if config['demoruns']:
                        while True:
                            demoenv=random.choice(evalenv)
                            a = SimulateAutomaticMode_PPO(demoenv, ppo_policy, t_suffix=False, entries=None)
                            if a == 'Q': break

                    result = evaluate_lstm_ppo(logdir=logdir_, config=config, env=evalenv, ppo_policy=ppo_policy, eval_subdir=evalName, max_num_nodes=world_dict[world_name][0])
                    num_unique_graphs, num_graph_instances, avg_return, success_rate = result
                    evalResults[evalName]['num_graphs.........'].append(num_unique_graphs)
                    evalResults[evalName]['num_graph_instances'].append(num_graph_instances)
                    evalResults[evalName]['avg_return.........'].append(avg_return)
                    evalResults[evalName]['success_rate.......'].append(success_rate)

            for ename, results in evalResults.items():
                OF = open(config['logdir']+'/Eval_det'+str(tp['eval_deterministic'])[0]+'_Results_over_seeds_'+ename+'.txt', 'w')
                def printing(text):
                    print(text)
                    OF.write(text + "\n")
                np.set_printoptions(formatter={'float':"{0:0.3f}".format})
                printing('Results over seeds for evaluation on '+ename+'\n')
                for category,values in results.items():
                    printing(category)
                    printing('  avg over seeds: '+str(np.mean(values)))
                    printing('  std over seeds: '+str(np.std(values)))
                    printing('  per seed: '+str(np.array(values))+'\n')

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
    parser.add_argument('--num_step', default=-1, type=int)
    parser.add_argument('--checkpoint_frequency', default=-1, type=int)
    parser.add_argument('--eval_deter', type=lambda s: s.lower() in ['true', 't', 'yes', '1'],default=True)
    parser.add_argument('--type_obs_wrap', default='obs_flat', type=str)
    parser.add_argument('--test_heur', type=lambda s: s.lower() in ['true', 't', 'yes', '1'],default=False)
    args=parser.parse_args()
    main(args)
