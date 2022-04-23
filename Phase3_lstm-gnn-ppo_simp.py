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
from modules.ppo.helpfuncs import CreateEnv, evaluate_lstm_ppo
from modules.rl.rl_utils import GetFullCoverageSample
from modules.rl.rl_policy import LSTM_GNN_PPO_Policy, LSTM_GNN_PPO_EMB_Policy, LSTM_GNN_PPO_EMB_Policy_simp
from modules.rl.rl_utils import EvaluatePolicy
from modules.sim.simdata_utils import SimulateAutomaticMode_PPO, SimulateInteractiveMode_PPO
from modules.ppo.ppo_wrappers import PPO_ActWrapper, PPO_ObsFlatWrapper
from modules.ppo.models_basic_lstm import PPO_GNN_LSTM, PPO_GNN_Dual_LSTM
#torch.set_num_threads(1) # Max #threads for torch to avoid inefficient util of cpu cores.


def main(args):
    config, hp, tp = GetConfigs(args, suffix='simp')   
    
    if config['train']:
        senv, env_all_train_list = ConstructTrainSet(config, apply_wrappers=True, type_obs_wrap='Flat', remove_paths=False, tset=config['train_on']) #TODO check

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
            #tp["workspace_path"]=logdir_

            if config['lstm_type'] in ['None', 'EMB']:
                model = PPO_GNN_LSTM(senv, config, hp, tp)
            elif config['lstm_type'] == 'Dual':
                model = PPO_GNN_Dual_LSTM(senv, config, hp, tp)
            if seed == config['seed0']: WriteModelParamsToFile(config, model)
            score = model.learn(senv)

    if config['test']:
        evalResults={}
        world_dict={ # [max_nodes,max_edges]
            #'Manhattan5x5_DuplicateSetB':[25,300],
            #'Manhattan3x3_WalkAround':[9,300],
            #'MetroU3_e1t31_FixedEscapeInit':[33, 300],
            # 'full_solvable_3x3subs':[9,33],
            # 'MemoryTaskU1':[8,16],
            'Manhattan5x5_FixedEscapeInit':[25,105],
            # 'Manhattan5x5_VariableEscapeInit':[25,105],
            # 'MetroU3_e17tborder_FixedEscapeInit':[33,300],
            # 'MetroU3_e17tborder_VariableEscapeInit':[33,300],
            # 'NWB_ROT_FixedEscapeInit':[2602,7300],
            # 'NWB_ROT_VariableEscapeInit':[2602,7300],
            # 'NWB_test_FixedEscapeInit':[975,4000],
            # 'NWB_test_VariableEscapeInit':[975,4000],
            # 'NWB_UTR_FixedEscapeInit':[1182,4000],
            # 'NWB_UTR_VariableEscapeInit':[1182,4000],
            # 'SparseManhattan5x5':[25,105],
            }
        #obs_mask='None'
        #obs_rate=1.
        #obs_mask='prob_per_u'
        #for obs_mask, obs_rate in zip(['None','prob_per_u','prob_per_u','prob_per_u'],[1.,.8,.6,.4]):
        for obs_mask, obs_rate in zip(['None'],[1.]):
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
                    env = CreateEnv(world_name, max_nodes=world_dict[world_name][0], max_edges = world_dict[world_name][1], nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=None, apply_wrappers=True, obs_mask=obs_mask, obs_rate=obs_rate)
                    evalenv=[env]
                hp.max_possible_nodes = world_dict[world_name][0]#env.max_possible_num_nodes
                hp.max_possible_edges = world_dict[world_name][1]#env.max_possible_num_edges

                evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
                for seed in config['seedrange']:
                    logdir_=config['logdir']+'/SEED'+str(seed)
                    tp["base_checkpoint_path"]=f"{logdir_}/checkpoints/"
                    try:
                        assert os.path.exists(tp['base_checkpoint_path'])
                    except:
                        continue
                    fname=tp['base_checkpoint_path']+'best_model.tar'
                    checkpoint = torch.load(fname)
                    if config['lstm_type'] in ['None', 'EMB']:
                        ppo_model = PPO_GNN_LSTM(env, config, hp, tp)
                    elif config['lstm_type'] == 'Dual':
                        ppo_model = PPO_GNN_Dual_LSTM(env, config, hp, tp)
                    ppo_model.load_state_dict(checkpoint['weights'])

                    if config['lstm_type'] == 'EMB':
                        ppo_policy = LSTM_GNN_PPO_EMB_Policy_simp(None, ppo_model, deterministic=tp['eval_deterministic'])
                    else:                   
                        pass
                        #ppo_policy = LSTM_GNN_PPO_Policy(env, ppo_model, deterministic=tp['eval_deterministic'])

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
    parser.add_argument('--eval_deter', type=lambda s: s.lower() in ['true', 't', 'yes', '1'],default=True)
    args=parser.parse_args()
    main(args)
