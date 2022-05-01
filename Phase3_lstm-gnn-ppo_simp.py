import os
import argparse
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from modules.dqn.dqn_utils import seed_everything
from modules.sim.graph_factory import GetWorldSet
import modules.gnn.nfm_gen
from modules.gnn.construct_trainsets import ConstructTrainSet
from modules.ppo.ppo_custom import WriteTrainParamsToFile, WriteModelParamsToFile, GetConfigs
from modules.ppo.helpfuncs import CreateEnv, CreateEnvFS, evaluate_lstm_ppo
from modules.rl.rl_utils import GetFullCoverageSample
from modules.rl.rl_policy import ColllisionRiskAvoidancePolicy, LSTM_GNN_PPO_Single_Policy_simp, LSTM_GNN_PPO_Dual_Policy_simp
from modules.rl.rl_utils import EvaluatePolicy
from modules.sim.simdata_utils import SimulateAutomaticMode_PPO, SimulateInteractiveMode_PPO
from modules.ppo.ppo_wrappers import PPO_ActWrapper, PPO_ObsBasicDictWrapper
from modules.ppo.models_basic_lstm import PPO_GNN_Single_LSTM, PPO_GNN_Dual_LSTM

def get_last_checkpoint_filename(tp):
    """
    Determine latest checkpoint iteration.
    """
    if os.path.isdir(tp['base_checkpoint_path']):
        max_checkpoint_iteration_list = [dirname.split('_')[-1].split('.')[0] for dirname in os.listdir(tp['base_checkpoint_path'])]
        max_checkpoint_iteration_list = [int(n) for n in max_checkpoint_iteration_list if n.isdigit()]
        if max_checkpoint_iteration_list == []:
            return None, 0, -1e6
        max_checkpoint_iteration = max(max_checkpoint_iteration_list)
        fname = tp['base_checkpoint_path'] + 'model_' + str(max_checkpoint_iteration) + '.tar'
        it = max_checkpoint_iteration
        OF = open(tp['base_checkpoint_path'] + 'model_best_save_history.txt', 'r')
        lines = OF.readlines()
        reslist = [float(e.split('res:')[1].split('\n')[0])*.99 for e in lines]
        best_result = max(reslist)
    else:
        return None, 0, -1e6
    return fname, it, best_result

def main(args):
    config, hp, tp = GetConfigs(args, suffix='simp')   
    
    ##### TRAIN FUNCTION #####
    if config['train']:
        senv, env_all_train_list = ConstructTrainSet(config, apply_wrappers=True, type_obs_wrap=config['type_obs_wrap'], remove_paths=False, tset=config['train_on']) #TODO check
        hp.node_dim = env_all_train_list[0].F

        if config['demoruns']:
            while True:
                a = SimulateInteractiveMode_PPO(senv, filesave_with_time_suffix=False)
                if a == 'Q': break
        
        WriteTrainParamsToFile(config,hp,tp)
        for seed in config['seedrange']:
            seed_everything(seed)
            senv.seed(seed)
            logdir_=config['logdir']+'/SEED'+str(seed)
            tp['writer'] = SummaryWriter(log_dir=f"{logdir_}/logs")
            tp["base_checkpoint_path"]=f"{logdir_}/checkpoints/"
            tp["seed_path"]=logdir_

            if config['lstm_type'] in ['None', 'EMB', 'FE']:
                model = PPO_GNN_Single_LSTM(config, hp, tp)
            elif config['lstm_type'] in ['Dual','DualCC']:
                model = PPO_GNN_Dual_LSTM(config, hp, tp)
            if seed == config['seed0']: WriteModelParamsToFile(config, model)
            last_checkpoint, it0, best_result = get_last_checkpoint_filename(tp)
            if last_checkpoint is not None:
                cp = torch.load(last_checkpoint)
                model.load_state_dict(cp['weights'])
                model.optimizer.load_state_dict(cp['optimizer'])
                print(f"Loaded model from {last_checkpoint}")
                print('Iteration:', it0, 'best_result:', best_result)
            try:
                score = model.learn(senv, it0, best_result)
            except:
                continue

    ##### EVALUATION FUNCTION FOR TRAINED POLICY for full test dataset #####
    if config['eval']:
        evalResults={}
        evalName='Trainset'
        senv, env_all_train_list = ConstructTrainSet(config, apply_wrappers=True, type_obs_wrap='BasicDict', remove_paths=False, tset=config['train_on']) #TODO check
        evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
        for seed in config['seedrange']:
            evalResults = GenerateResults(seed, env_all_train_list, evalName, evalResults, config, hp, tp, maxnodes=senv.max_possible_num_nodes)
        SaveResults(evalResults, config, tp)

    ##### TEST FUNCTION FOR TRAINED POLICY #####
    if config['test']:
        evalResults={}
        world_dict = SelectTestWorlds()
        obs_evalmasks = ['None','prob_per_u_test','prob_per_u_test','prob_per_u_test','prob_per_u_test','prob_per_u_test'] # ['None']['prob_per_u']
        obs_evalrates = [1.0,0.9,.8,.7,.6,.5]    # [1.][0.8]
        for obs_mask, obs_rate in zip(obs_evalmasks, obs_evalrates):
            for world_name in world_dict.keys():
                evalName=world_name+'_obs'+obs_mask+'_evaldet'+str(tp['eval_deterministic'])[0]
                if obs_mask != 'None': evalName += str(obs_rate)
                if world_name == "full_solvable_3x3subs":
                    evalenv = CreateEnvFS(config, obs_mask, obs_rate, max_nodes=world_dict[world_name][0], max_edges=world_dict[world_name][1])
                else:
                    env = CreateEnv(world_name, max_nodes=world_dict[world_name][0], max_edges = world_dict[world_name][1], nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=None, apply_wrappers=True, type_obs_wrap='BasicDict', obs_mask=obs_mask, obs_rate=obs_rate)
                    evalenv=[env]

                evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
                for seed in config['seedrange']:
                    evalResults = GenerateResults(seed, evalenv, evalName, evalResults, config, hp, tp, world_dict[world_name][0])

                SaveResults(evalResults, config, tp)

    ##### EVALUATION FUNCTION FOR HEURSTIC POLICY for selected dataset #####
    if config['test_heur']:
        l=config['logdir'].find(config['train_on'])
        r=config['logdir'].find(config['nfm_func'])
        to=len(config['train_on'])
        config['logdir']=config['logdir'][:(l+to+1)]+config['logdir'][r:]
        config['rootdir']=config['rootdir'][:l+to]
        evalResults={}
        world_dict = SelectTestWorlds()
        obs_evalmasks = ['prob_per_u_test','prob_per_u_test']#,'prob_per_u_test','prob_per_u_test','prob_per_u_test'] # ['None']['prob_per_u']
        obs_evalrates = [0.6,0.5]#,0.7,0.6,0.5]    # [1.][0.8]
        for obs_mask, obs_rate in zip(obs_evalmasks, obs_evalrates):
            for world_name in world_dict.keys():
                evalName=world_name+'_obs'+obs_mask
                if obs_mask != 'None': evalName += str(obs_rate)
                if world_name == "full_solvable_3x3subs":
                    evalenv = CreateEnvFS(config, obs_mask, obs_rate, max_nodes=world_dict[world_name][0], max_edges=world_dict[world_name][1])
                else:
                    env = CreateEnv(world_name, max_nodes=world_dict[world_name][0], max_edges = world_dict[world_name][1], nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=None, apply_wrappers=True, type_obs_wrap='BasicDict', obs_mask=obs_mask, obs_rate=obs_rate)
                    assert config['type_obs_wrap']=='BasicDict'
                    evalenv=[env]

                evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
                
                evalResults = GenerateResultsHeur(evalenv, evalName, evalResults, config, hp, tp, world_dict[world_name][0])

                SaveResults(evalResults, config, tp)


def SelectTestWorlds():
    world_dict={ # [max_nodes,max_edges]
            #'Manhattan5x5_DuplicateSetB':[25,300],
            #'Manhattan3x3_WalkAround':[9,33],
            #'MetroU3_e1t31_FixedEscapeInit':[33, 300],
            #'MemoryTaskU1':[8,16],
            'full_solvable_3x3subs':[9,33],
            'Manhattan5x5_FixedEscapeInit':[25,105],
            #'Manhattan5x5_FixedEscapeInit2':[25,105],
            'Manhattan5x5_VariableEscapeInit':[25,105],
            'MetroU3_e17tborder_FixedEscapeInit':[33,300],
            'MetroU3_e17tborder_VariableEscapeInit':[33,300],
            'NWB_ROT_FixedEscapeInit':[2602,7300],
            'NWB_ROT_VariableEscapeInit':[2602,7300],
            #'NWB_test_FixedEscapeInit':[975,1425],
            #'NWB_test_FixedEscapeInit2':[975,1425],
            'NWB_test_VariableEscapeInit':[975,1425],
            'NWB_UTR_FixedEscapeInit':[1182,4000],
            'NWB_UTR_VariableEscapeInit':[1182,4000],
            #'SparseManhattan5x5':[25,105],
            }
    return world_dict

def GenerateResults(seed, evalenv, evalName, evalResults, config, hp, tp, maxnodes):                    
    logdir_=config['logdir']+'/SEED'+str(seed)
    tp["base_checkpoint_path"]=f"{logdir_}/checkpoints/"
    try:
        assert os.path.exists(tp['base_checkpoint_path'])
    except:
        return evalResults
    fname=tp['base_checkpoint_path']+'best_model.tar'
    checkpoint = torch.load(fname)
    
    if config['lstm_type'] in ['None', 'EMB','FE']:
        ppo_model = PPO_GNN_Single_LSTM(config, hp, tp)
    elif config['lstm_type'] in ['Dual','DualCC']:
        ppo_model = PPO_GNN_Dual_LSTM(config, hp, tp)
    ppo_model.load_state_dict(checkpoint['weights'])
    print('Loaded model from', fname)
    
    if config['lstm_type'] in  ['EMB','FE','None']:
        ppo_policy = LSTM_GNN_PPO_Single_Policy_simp(ppo_model, deterministic=tp['eval_deterministic'])
    elif config['lstm_type'] in ['Dual','DualCC']:                   
        ppo_policy = LSTM_GNN_PPO_Dual_Policy_simp(ppo_model, deterministic=tp['eval_deterministic'])
    else: assert False

    if config['demoruns']:
        while True:
            demoenv=random.choice(evalenv)
            a = SimulateAutomaticMode_PPO(demoenv, ppo_policy, t_suffix=False, entries=None)
            if a == 'Q': break

    result = evaluate_lstm_ppo(logdir=logdir_, config=config, env=evalenv, ppo_policy=ppo_policy, eval_subdir=evalName, max_num_nodes=maxnodes)
    num_unique_graphs, num_graph_instances, avg_return, success_rate = result
    evalResults[evalName]['num_graphs.........'].append(num_unique_graphs)
    evalResults[evalName]['num_graph_instances'].append(num_graph_instances)
    evalResults[evalName]['avg_return.........'].append(avg_return)
    evalResults[evalName]['success_rate.......'].append(success_rate)

    return evalResults

def GenerateResultsHeur(evalenv, evalName, evalResults, config, hp, tp, maxnodes):                    
    logdir_=config['logdir']
    
    ppo_policy = ColllisionRiskAvoidancePolicy(evalenv[0])

    if config['demoruns']:
        while True:
            demoenv=random.choice(evalenv)
            a = SimulateAutomaticMode_PPO(demoenv, ppo_policy, t_suffix=False, entries=None)
            if a == 'Q': break

    result = evaluate_lstm_ppo(logdir=logdir_, config=config, env=evalenv, ppo_policy=ppo_policy, eval_subdir=evalName, max_num_nodes=maxnodes)
    num_unique_graphs, num_graph_instances, avg_return, success_rate = result
    evalResults[evalName]['num_graphs.........'].append(num_unique_graphs)
    evalResults[evalName]['num_graph_instances'].append(num_graph_instances)
    evalResults[evalName]['avg_return.........'].append(avg_return)
    evalResults[evalName]['success_rate.......'].append(success_rate)

    return evalResults


def SaveResults(evalResults, config, tp):
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
    parser.add_argument('--num_step', default=10000, type=int)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--recurrent_seq_len', default=2, type=int)
    parser.add_argument('--parallel_rollouts', default=1, type=int)
    parser.add_argument('--rollout_steps', default=100, type=int)
    parser.add_argument('--patience', default=500, type=int)
    parser.add_argument('--checkpoint_frequency', default=5000, type=int)
    parser.add_argument('--obs_mask', default='None', type=str, help='U obervation masking type', choices=['None','freq','prob','prob_per_u','mix'])
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
    parser.add_argument('--test_heur', type=lambda s: s.lower() in ['true', 't', 'yes', '1'],default=False)       
    parser.add_argument('--num_seeds', default=5, type=int)
    parser.add_argument('--seed0', default=10, type=int)
    parser.add_argument('--demoruns', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--eval_deter', type=lambda s: s.lower() in ['true', 't', 'yes', '1'],default=True)
    parser.add_argument('--type_obs_wrap', default='', type=str)
    args=parser.parse_args()
    main(args)
