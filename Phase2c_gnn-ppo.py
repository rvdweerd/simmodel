import argparse
import numpy as np
import torch
import random
from sb3_contrib import MaskablePPO
from modules.sim.graph_factory import GetWorldSet
from modules.gnn.comb_opt import evaluate_spath_heuristic
from modules.rl.rl_utils import EvaluatePolicy, print_parameters, GetFullCoverageSample, NpWrapper
from modules.ppo.helpfuncs import get_super_env, CreateEnv, eval_simple, evaluate_ppo#, get_logdirs
from modules.ppo.callbacks_sb3 import SimpleCallback, TestCallBack
from stable_baselines3.common.callbacks import EvalCallback
from modules.ppo.models_sb3_s2v import s2v_ActorCriticPolicy, Struc2VecExtractor, DeployablePPOPolicy
from modules.ppo.models_sb3_gat2 import Gat2_ActorCriticPolicy, Gat2Extractor, DeployablePPOPolicy_gat2
from modules.rl.rl_policy import ActionMaskedPolicySB3_PPO
from modules.rl.environments import SuperEnv
import modules.gnn.nfm_gen
from modules.gnn.construct_trainsets import ConstructTrainSet, get_train_configs
from modules.sim.simdata_utils import SimulateInteractiveMode, SimulateInteractiveMode_PPO, SimulateAutomaticMode_PPO
from Phase2c_eval import ManualEval
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def GetConfig(args):
    config={}
    config['train_on'] = args.train_on
    config['max_nodes'] = args.max_nodes
    config['remove_paths'] = args.pursuit == 'Uoff'
    assert args.pursuit in ['Uoff','Uon']
    config['reject_u_duplicates'] = False
    config['solve_select'] = args.solve_select # only solvable worlds (so best achievable performance is 100%)
    config['nfm_func']     = args.nfm_func
    config['edge_blocking']= args.edge_blocking
    config['node_dim']     = modules.gnn.nfm_gen.nfm_funcs[args.nfm_func].F
    config['demoruns']     = args.demoruns
    config['qnet']         = args.qnet
    config['norm_agg']     = args.norm_agg
    config['emb_dim']      = args.emb_dim
    config['emb_iter_T']   = args.emb_itT 
    config['optim_target'] = args.optim_target
    #config['num_episodes'] = args.num_epi 
    #config['memory_size']  = args.mem_size 
    config['num_step']      = args.num_step
    config['obs_mask']      = "None"
    config['obs_rate']      = 1.0
    #config['bsize']        = 32        
    #config['gamma']        = .9        
    #config['lr_init']      = 1e-3      
    #config['lr_decay']     = 0.99999    
    #config['tau']          = args.tau  # num grad steps for each target network update
    #config['eps_0']        = 1.        
    #config['eps_min']      = 0.1
    #config['epi_min']      = .9        # reach eps_min at % of episodes # .9
    #config['eps_decay']    = 1 - np.exp(np.log(config['eps_min'])/(config['epi_min']*config['num_episodes']))
    config['rootdir']='./results/results_Phase2/Pathfinding/ppo/'+ \
                                config['train_on']+'_'+args.pursuit+'/'+ \
                                config['solve_select']+'_edgeblock'+str(config['edge_blocking'])+'/'+\
                                config['qnet']+'_normagg'+str(config['norm_agg'])
    config['logdir']        = config['rootdir'] + '/' +\
                                config['nfm_func']+'/'+ \
                                'emb'+str(config['emb_dim']) + \
                                '_itT'+str(config['emb_iter_T']) + \
                                '_nstep'+str(config['num_step'])
    config['seed0']=args.seed0
    config['numseeds']=args.num_seeds
    config['seedrange']=range(config['seed0'], config['seed0']+config['numseeds'])
    print('seedrange')
    for i in config['seedrange']: print(i)
    return config

def main(args):
    config=GetConfig(args)
    #env_train   = config['env_train']
    if args.train: #or args.eval:
        senv, env_all_train_list = ConstructTrainSet(config, apply_wrappers=True, type_obs_wrap='Dict', remove_paths=config['remove_paths'], tset=config['train_on']) #TODO check
        #env_all_train = [senv]
        if config['demoruns']:
            while True:
                a = SimulateInteractiveMode_PPO(senv, filesave_with_time_suffix=False)
                if a == 'Q': break

        assert config['node_dim'] == senv.F

        policy_kwargs = dict(
            features_extractor_class = Struc2VecExtractor if config['qnet']=='s2v' else Gat2Extractor,
            features_extractor_kwargs = dict(emb_dim=config['emb_dim'], emb_iter_T=config['emb_iter_T'], node_dim=config['node_dim']),#, num_nodes=MAX_NODES),
            # NOTE: FOR THIS TO WORK, NEED TO ADJUST sb3 policies.py
            #           def proba_distribution_net(self, latent_dim: int) -> nn.Module:
            #       to create a linear layer that maps to 1-dim instead of self.action_dim
            #       reason: our model is invariant to the action space (number of nodes in the graph) 
        )

        for seed in config['seedrange']:
            logdir_ = config['logdir']+'/SEED'+str(seed)
            if config['qnet'] not in ['s2v','gat2']: assert False
            model = MaskablePPO(policy = s2v_ActorCriticPolicy if config['qnet']=='s2v' else Gat2_ActorCriticPolicy, \
                env = senv, \
                #learning_rate=1e-5, #3e-4
                #n_steps = 1024, #2048
                seed=seed, \
                #n_epochs = 10,#10
                batch_size=32,#5, #64
                #clip_range=0.1,\    
                #max_grad_norm=0.1,\
                policy_kwargs = policy_kwargs, verbose=2, tensorboard_log=logdir_+"/tb/")

            # senv_eval = Monitor(senv)
            # senv_eval = DummyVecEnv([lambda: senv_eval])
            # eval_callback = EvalCallback(senv_eval, best_model_save_path=logdir_+'/saved_models/',
            #                             log_path=logdir_+'/logs/', eval_freq=2000,
            #                             deterministic=True, render=False, verbose=2)    

            test_callback = TestCallBack(verbose=2, logdir=logdir_+'/saved_models')
            print_parameters(model.policy)
            model.learn(total_timesteps = config['num_step'], callback=[test_callback])#,eval_callback]) #,wandb_callback])
            # run.finish()
            model.save(logdir_+"/saved_models/model_last")
            model.policy.save(logdir_+"/saved_models/policy_last")    
    
    if args.test:
        # world_list=['Manhattan3x3_WalkAround',
        #             'MetroU3_e1t31_FixedEscapeInit', 
        #             'NWB_test_VariableEscapeInit']
        # node_maxims = [9,33,975]
        # var_targets=[ None, [1,1], None]
        # eval_names =  [ 'Manhattan3x3_WalkAround'
        #                 'MetroU0_e1t31_vartarget_eval', 
        #                 'NWB_VarialeEscapeInit_eval' ]
        # eval_nums = [1, 1000,200]
        evalResults={}
        world_dict={ # [max_nodes,max_edges]
            #'Manhattan5x5_DuplicateSetB':[25,300],
            #'Manhattan3x3_WalkAround':[9,300],
            #'MetroU3_e1t31_FixedEscapeInit':[33, 300],
            #'full_solvable_3x3subs':[9,300],
            #'Manhattan5x5_FixedEscapeInit':[25,300],
            #'Manhattan5x5_VariableEscapeInit':[25,300],
            #'MetroU3_e17tborder_FixedEscapeInit':[33,300],
            #'MetroU3_e17tborder_VariableEscapeInit':[33,300],
            'NWB_ROT_FixedEscapeInit':[2602,7300],
            'NWB_ROT_VariableEscapeInit':[2602,7300],
            'NWB_test_FixedEscapeInit':[975,4000],
            'NWB_test_VariableEscapeInit':[975,4000],
            'NWB_UTR_FixedEscapeInit':[1182,4000],
            'NWB_UTR_VariableEscapeInit':[1182,4000],
            # 'SparseManhattan5x5':[25,300],
            }
        #for world_name, node_maxim, var_target, eval_name, eval_num in zip(world_list, node_maxims, var_targets, eval_names, eval_nums):
        #custom_env = CreateEnv('MetroU3_e1t31_FixedEscapeInit',max_nodes=33,nfm_func_name =config['nfm_func'],var_targets=[1,1], remove_world_pool=True, apply_wrappers=True)
        #evalName='MetroU0_e1t31_vartarget_eval'
        for world_name in world_dict.keys():
            evalName=world_name
            if world_name == "full_solvable_3x3subs":
                Etest=[0,1,2,3,4,5,6,7,8,9,10]
                Utest=[1,2,3]
                evalenv, _, _, _  = GetWorldSet('etUte0U0', 'nfm', U=Utest, E=Etest, edge_blocking=config['edge_blocking'], solve_select=config['solve_select'], reject_duplicates=False, nfm_func=modules.gnn.nfm_gen.nfm_funcs[config['nfm_func']], apply_wrappers=True, maxnodes=world_dict[world_name][0], maxedges=world_dict[world_name][1])
            else:
                env = CreateEnv(world_name, max_nodes=world_dict[world_name][0], max_edges = world_dict[world_name][1], nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=config['remove_paths'], apply_wrappers=True, type_obs_wrap='Dict')
                evalenv=[env]

            if config['demoruns']:
                saved_model = MaskablePPO.load(config['logdir']+'/SEED'+str(config['seed0'])+"/saved_models/model_best")
                if config['qnet']=='s2v':
                    #saved_policy = s2v_ActorCriticPolicy.load(config['logdir']+'/SEED'+str(config['seed0'])+"/saved_models/policy_last")
                    saved_policy_deployable=DeployablePPOPolicy(evalenv[0], saved_model.policy)
                    ppo_policy = ActionMaskedPolicySB3_PPO(saved_policy_deployable, deterministic=True)
                else:
                    #saved_policy = Gat2_ActorCriticPolicy.load(config['logdir']+'/SEED'+str(config['seed0'])+"/saved_models/policy_last")
                    saved_policy_deployable=DeployablePPOPolicy_gat2(evalenv[0], saved_model.policy, max_num_nodes=world_dict[world_name][0])
                    ppo_policy = ActionMaskedPolicySB3_PPO(saved_policy_deployable, deterministic=True)
                while True:
                    entries=None#[5012,218,3903]
                    demo_env = random.choice(evalenv)
                    a = SimulateAutomaticMode_PPO(demo_env, ppo_policy, t_suffix=False, entries=entries)
                    if a == 'Q': break

            evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
            for seed in config['seedrange']:
                result = evaluate_ppo(logdir=config['logdir']+'/SEED'+str(seed), config=config, env=evalenv, eval_subdir=evalName, max_num_nodes=world_dict[world_name][0])
                num_unique_graphs, num_graph_instances, avg_return, success_rate = result
                evalResults[evalName]['num_graphs.........'].append(num_unique_graphs)
                evalResults[evalName]['num_graph_instances'].append(num_graph_instances)
                evalResults[evalName]['avg_return.........'].append(avg_return)
                evalResults[evalName]['success_rate.......'].append(success_rate)

        for ename, results in evalResults.items():
            OF = open(config['logdir']+'/Results_over_seeds_'+ename+'.txt', 'w')
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

    if args.eval:
        ManualEval(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--train_on', default='None', type=str)
    parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('--emb_itT', default=2, type=int)
    parser.add_argument('--nfm_func', default='NFM_ev_ec_t', type=str)
    parser.add_argument('--qnet', default='None', type=str)
    parser.add_argument('--norm_agg', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--optim_target', default='None', type=str)
    parser.add_argument('--num_step', default=10000, type=int)
    parser.add_argument('--train', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--eval', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])   
    parser.add_argument('--test', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])       
    parser.add_argument('--num_seeds', default=5, type=int)
    parser.add_argument('--seed0', default=10, type=int)
    parser.add_argument('--solve_select', default='solvable', type=str)
    parser.add_argument('--edge_blocking', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--max_nodes', default=9, type=int)
    parser.add_argument('--demoruns', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--pursuit', default='Uon', type=str)
    args=parser.parse_args()
    main(args)