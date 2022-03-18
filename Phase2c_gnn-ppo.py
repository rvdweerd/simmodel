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
from modules.ppo.models_sb3 import s2v_ActorCriticPolicy, Struc2Vec, DeployablePPOPolicy
from modules.rl.rl_policy import ActionMaskedPolicySB3_PPO
from modules.rl.environments import SuperEnv
import modules.gnn.nfm_gen
from modules.gnn.construct_trainsets import ConstructTrainSet, get_train_configs
from modules.sim.simdata_utils import SimulateInteractiveMode, SimulateAutomaticMode_PPO
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
    config['num_step']     = args.num_step
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
    if args.train or args.eval:
        senv, env_all_train_list = ConstructTrainSet(config, apply_wrappers=True, remove_paths=config['remove_paths'], tset=config['train_on']) #TODO check
        #env_all_train = [senv]
        if config['demoruns']:
            while True:
                a = SimulateInteractiveMode(senv, filesave_with_time_suffix=False)
                if a == 'Q': break

    if args.train:
        #obs=senv.reset()
        assert config['node_dim'] == senv.F

        policy_kwargs = dict(
            features_extractor_class = Struc2Vec,
            features_extractor_kwargs = dict(emb_dim=config['emb_dim'], emb_iter_T=config['emb_iter_T'], node_dim=config['node_dim']),#, num_nodes=MAX_NODES),
            # NOTE: FOR THIS TO WORK, NEED TO ADJUST sb3 policies.py
            #           def proba_distribution_net(self, latent_dim: int) -> nn.Module:
            #       to create a linear layer that maps to 1-dim instead of self.action_dim
            #       reason: our model is invariant to the action space (number of nodes in the graph) 
        )

        for seed in config['seedrange']:
            logdir_ = config['logdir']+'/SEED'+str(seed)

            model = MaskablePPO(s2v_ActorCriticPolicy, senv, \
                #learning_rate=1e-4,\
                seed=seed, \
                #batch_size=128, \
                #clip_range=0.1,\    
                #max_grad_norm=0.1,\
                policy_kwargs = policy_kwargs, verbose=2, tensorboard_log=logdir_+"/tb/")

            print_parameters(model.policy)
            model.learn(total_timesteps = config['num_step'], callback=[TestCallBack()])#,eval_callback]) #,wandb_callback])
            # run.finish()
            model.save(logdir_+"/saved_models/model_last")
            model.policy.save(logdir_+"/saved_models/policy_last")    

    if args.eval:
        evalResults={}
        test_heuristics              = False
        test_full_trainset           = False
        test_full_solvable_3x3subs   = False
        test_all_solvable_3x3segments= False
        test_other_worlds            = False
        #test_specific_world          = True

        if test_heuristics:
            # Evaluate using shortest path heuristics on the full trainset
            evaluate_spath_heuristic(logdir=config['rootdir']+'/heur/spath', config=config, env_all=env_train)
        
        if test_full_trainset:
            # Evaluate on the full training set
            evalName='trainset_eval'
            evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 

            for seed in config['seedrange']:
                saved_policy = s2v_ActorCriticPolicy.load(config['logdir']+'/SEED'+str(seed)+"/saved_models/policy_last")
                policy = ActionMaskedPolicySB3_PPO(saved_policy, deterministic=True)

                result = evaluate_ppo(logdir=config['logdir']+'/SEED'+str(seed), policy=policy, config=config, env=env_train, eval_subdir=evalName, n_eval=5000)
                num_unique_graphs, num_graph_instances, avg_return, success_rate = result
                evalResults[evalName]['num_graphs.........'].append(num_unique_graphs)
                evalResults[evalName]['num_graph_instances'].append(num_graph_instances)
                evalResults[evalName]['avg_return.........'].append(avg_return)
                evalResults[evalName]['success_rate.......'].append(success_rate)

        if test_full_solvable_3x3subs:
            # Evaluate on the full evaluation set
            evalName='3x3full_testset_eval'
            evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
            Utest=[0,1,2,3]
            Etest=[0,1,2,3,4,5,6,7,8,9]
            config['solve_select']='solvable'
            _, env_all_test = get_super_env(Uselected=Utest, Eselected=Etest, config=config)
         
            for seed in config['seedrange']:
                saved_policy = s2v_ActorCriticPolicy.load(config['logdir']+'/SEED'+str(seed)+"/saved_models/policy_last")
                policy = ActionMaskedPolicySB3_PPO(saved_policy, deterministic=True)

                result = evaluate_ppo(logdir=config['logdir']+'/SEED'+str(seed), policy=policy, config=config, env=env_all_test, eval_subdir=evalName)
                num_unique_graphs, num_graph_instances, avg_return, success_rate = result
                evalResults[evalName]['num_graphs.........'].append(num_unique_graphs)
                evalResults[evalName]['num_graph_instances'].append(num_graph_instances)
                evalResults[evalName]['avg_return.........'].append(avg_return)
                evalResults[evalName]['success_rate.......'].append(success_rate)
        
        if test_all_solvable_3x3segments:
            # Evaluate on each individual segment of the evaluation set
            Utest=[0,1,2,3]
            Etest=[0,1,2,3,4,5,6,7,8,9]
            evalName='graphsegments_eval'
            config['solve_select']='solvable'
            evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
            for seed in config['seedrange']:
                saved_policy = s2v_ActorCriticPolicy.load(config['logdir']+'/SEED'+str(seed)+"/saved_models/policy_last")
                policy = ActionMaskedPolicySB3_PPO(saved_policy, deterministic=True)
                success_matrix   =[]
                num_graphs_matrix=[]
                instances_matrix =[]
                returns_matrix   =[]
                for u in Utest:
                    success_vec   =[]
                    num_graphs_vec=[]
                    instances_vec =[]
                    returns_vec   =[]
                    for e in Etest:
                        _, env_all_test = get_super_env(Uselected=[u], Eselected=[e], config=config)
                        result = evaluate_ppo(logdir=config['logdir']+'/SEED'+str(seed), policy=policy, config=config, env=env_all_test, eval_subdir=evalName+'/runs/'+'E'+str(e)+'U'+str(u))
                        num_unique_graphs, num_graph_instances, avg_return, success_rate = result         
                        success_vec.append(success_rate)
                        num_graphs_vec.append(num_unique_graphs)
                        instances_vec.append(num_graph_instances)
                        returns_vec.append(avg_return)
                    success_matrix.append(success_vec)
                    num_graphs_matrix.append(num_graphs_vec)
                    instances_matrix.append(instances_vec)
                    returns_matrix.append(returns_vec)
                OF = open(config['logdir']+'/SEED'+str(seed)+'/'+evalName+'/success_matrix.txt', 'w')
                def printing(text):
                    print(text)
                    OF.write(text + "\n")    
                success_matrix = np.array(success_matrix)
                returns_matrix = np.array(returns_matrix)
                num_graphs_matrix = np.array(num_graphs_matrix)
                instances_matrix = np.array(instances_matrix)
                weighted_return = (returns_matrix * instances_matrix).sum() / instances_matrix.sum()
                weighted_success_rate = (success_matrix * instances_matrix).sum() / instances_matrix.sum()
                np.set_printoptions(formatter={'float':"{0:0.3f}".format})
                printing('success_matrix:')
                printing(str(success_matrix))
                printing('\nreturns_matrix:')
                printing(str(returns_matrix))
                printing('\nnum_graphs_matrix:')
                printing(str(num_graphs_matrix))
                printing('\ninstances_matrix:')
                printing(str(instances_matrix))
                printing('\nWeighted return: '+str(weighted_return))
                printing('Weighted success rate: '+str(weighted_success_rate))
                OF.close()
                evalResults[evalName]['num_graphs.........'].append(num_graphs_matrix.sum())
                evalResults[evalName]['num_graph_instances'].append(instances_matrix.sum())
                evalResults[evalName]['avg_return.........'].append(weighted_return)
                evalResults[evalName]['success_rate.......'].append(weighted_success_rate)

        #
        #   Evaluate learned model on another (out of distribution) graph
        #
        if test_other_worlds:
            world_names=[
                #'SparseManhattan5x5',
                #'MetroU3_e17tborder_VariableEscapeInit',
                'MetroU3_e17t31_FixedEscapeInit',
                #'Manhattan5x5_FixedEscapeInit',
                #'Manhattan5x5_VariableEscapeInit',
            ]
            state_repr='etUte0U0'
            state_enc='nfm'
            for world_name in world_names:
                evalName=world_name[:16]+'_eval'
                evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
                
                custom_env=CreateEnv(world_name=world_name, max_nodes=config['max_nodes'])
                #custom_env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
                #custom_env.redefine_nfm(nfm_func)
                for seed in config['seedrange']:
                    saved_policy = s2v_ActorCriticPolicy.load(config['logdir']+'/SEED'+str(seed)+"/saved_models/policy_last")
                    policy = ActionMaskedPolicySB3_PPO(saved_policy, deterministic=True)

                    result = evaluate_ppo(logdir=config['logdir']+'/SEED'+str(seed), policy=policy, config=config, env=[custom_env], eval_subdir=evalName)
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
        world_dict={
            #'Manhattan5x5_DuplicateSetB':25,
            #'Manhattan3x3_WalkAround':9,
            #'MetroU3_e1t31_FixedEscapeInit':33, 
            'full_solvable_3x3subs':9,
            'Manhattan5x5_FixedEscapeInit':25,
            'Manhattan5x5_VariableEscapeInit':25,
            'MetroU3_e17tborder_FixedEscapeInit':33,
            'MetroU3_e17tborder_VariableEscapeInit':33,
            'NWB_ROT_FixedEscapeInit':2602,
            'NWB_ROT_VariableEscapeInit':2602,
            'NWB_test_FixedEscapeInit':975,
            'NWB_test_VariableEscapeInit':975,
            'NWB_UTR_FixedEscapeInit':1182,
            'NWB_UTR_VariableEscapeInit':1182,            
            'SparseManhattan5x5':25,
            }
        #for world_name, node_maxim, var_target, eval_name, eval_num in zip(world_list, node_maxims, var_targets, eval_names, eval_nums):
        #custom_env = CreateEnv('MetroU3_e1t31_FixedEscapeInit',max_nodes=33,nfm_func_name =config['nfm_func'],var_targets=[1,1], remove_world_pool=True, apply_wrappers=True)
        #evalName='MetroU0_e1t31_vartarget_eval'
        for world_name in world_dict.keys():
            evalName=world_name
            if world_name == "full_solvable_3x3subs":
                Etest=[0,1,2,3,4,5,6,7,8,9,10]
                Utest=[1,2,3]
                evalenv, _, _, _  = GetWorldSet('etUte0U0', 'nfm', U=Utest, E=Etest, edge_blocking=config['edge_blocking'], solve_select=config['solve_select'], reject_duplicates=False, nfm_func=modules.gnn.nfm_gen.nfm_funcs[config['nfm_func']], apply_wrappers=True, maxnodes=world_dict[world_name])
            else:
                env = CreateEnv(world_name, max_nodes=world_dict[world_name], nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=config['remove_paths'], apply_wrappers=True)
                evalenv=[env]

            if config['demoruns']:
                saved_model = MaskablePPO.load(config['logdir']+'/SEED'+str(config['seed0'])+"/saved_models/model_last")
                saved_policy = s2v_ActorCriticPolicy.load(config['logdir']+'/SEED'+str(config['seed0'])+"/saved_models/policy_last")
                saved_policy_deployable=DeployablePPOPolicy(env, saved_policy)
                ppo_policy = ActionMaskedPolicySB3_PPO(saved_policy_deployable, deterministic=True)
                while True:
                    entries=None#[5012,218,3903]
                    demo_env = random.choice(evalenv)
                    a = SimulateAutomaticMode_PPO(demo_env, ppo_policy, t_suffix=False, entries=entries)
                    if a == 'Q': break



            #custom_env = CreateEnv(world_name, max_nodes=node_maxim, nfm_func_name = config['nfm_func'], var_targets=var_target, remove_world_pool=True, apply_wrappers=True)
            #senv=SuperEnv([custom_env],{1:0},node_maxim,probs=[1])        
            #n_eval=eval_num
            evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
            for seed in config['seedrange']:
                result = evaluate_ppo(logdir=config['logdir']+'/SEED'+str(seed), config=config, env=evalenv, eval_subdir=evalName)
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