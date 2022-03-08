import argparse
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from modules.gnn.comb_opt import evaluate_spath_heuristic
from modules.rl.rl_utils import EvaluatePolicy, print_parameters, GetFullCoverageSample, NpWrapper
from modules.ppo.helpfuncs import get_super_env, CreateEnv, eval_simple, evaluate_ppo, get_logdirs
from modules.ppo.callbacks_sb3 import SimpleCallback, TestCallBack
from modules.ppo.models_sb3 import s2v_ActorCriticPolicy, Struc2Vec, DeployablePPOPolicy
from modules.rl.rl_policy import ActionMaskedPolicySB3_PPO
from modules.rl.environments import SuperEnv
from modules.sim.simdata_utils import SimulateInteractiveMode_PPO
from Phase2d_construct_sets import ConstructTrainSet, get_train_configs
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--train', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--eval', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])   
    parser.add_argument('--test', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])   


    args=parser.parse_args()
    run_name = args.run_name
    
    if args.train or args.eval:
        train_configs=get_train_configs(run_name,load_trainset=True)
    else:
        train_configs=get_train_configs(run_name,load_trainset=False)
    config=train_configs[run_name]
    train = args.train
    eval  = args.eval
    test = args.test

    MAX_NODES = config['max_nodes']
    EMB_DIM = config['emb_dim']
    EMB_ITER_T = config['emb_iter_T']
    TOTAL_TIME_STEPS = config['num_step']
    SEED0 = config['seed0']
    NUMSEEDS = config['numseeds']
    LOGDIR = config['logdir']
    NODE_DIM = config['node_dim']
    env_train = config['env_train']

    if train:
        obs=env_train.reset()
        assert NODE_DIM == env_train.F

        policy_kwargs = dict(
            features_extractor_class = Struc2Vec,
            features_extractor_kwargs = dict(emb_dim=EMB_DIM, emb_iter_T=EMB_ITER_T, node_dim=NODE_DIM),#, num_nodes=MAX_NODES),
            # NOTE: FOR THIS TO WORK, NEED TO ADJUST sb3 policies.py
            #           def proba_distribution_net(self, latent_dim: int) -> nn.Module:
            #       to create a linear layer that maps to 1-dim instead of self.action_dim
            #       reason: our model is invariant to the action space (number of nodes in the graph) 
        )

        for seed in range(SEED0, SEED0+NUMSEEDS):
            logdir_ = LOGDIR+'/SEED'+str(seed)


            model = MaskablePPO(s2v_ActorCriticPolicy, env_train, \
                #learning_rate=1e-4,\
                seed=seed, \
                #batch_size=128, \
                #clip_range=0.1,\    
                #max_grad_norm=0.1,\
                policy_kwargs = policy_kwargs, verbose=2, tensorboard_log=logdir_+"/tb/")

            print_parameters(model.policy)
            model.learn(total_timesteps = TOTAL_TIME_STEPS, callback=[TestCallBack()])#,eval_callback]) #,wandb_callback])
            # run.finish()
            model.save(logdir_+"/saved_models/model_last")
            model.policy.save(logdir_+"/saved_models/policy_last")    

    if eval:
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

            for seed in range(SEED0, SEED0+NUMSEEDS):
                saved_policy = s2v_ActorCriticPolicy.load(LOGDIR+'/SEED'+str(seed)+"/saved_models/policy_last")
                policy = ActionMaskedPolicySB3_PPO(saved_policy, deterministic=True)

                result = evaluate_ppo(logdir=LOGDIR+'/SEED'+str(seed), policy=policy, config=config, env=env_train, eval_subdir=evalName, n_eval=5000)
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
         
            for seed in range(SEED0, SEED0+NUMSEEDS):
                saved_policy = s2v_ActorCriticPolicy.load(LOGDIR+'/SEED'+str(seed)+"/saved_models/policy_last")
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
            for seed in range(SEED0, SEED0+NUMSEEDS):
                saved_policy = s2v_ActorCriticPolicy.load(LOGDIR+'/SEED'+str(seed)+"/saved_models/policy_last")
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
                for seed in range(SEED0, SEED0+NUMSEEDS):
                    saved_policy = s2v_ActorCriticPolicy.load(LOGDIR+'/SEED'+str(seed)+"/saved_models/policy_last")
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
    
    if test:
        world_list=['MetroU3_e1t31_FixedEscapeInit', 
                    'NWB_test_VariableEscapeInit']
        node_maxims = [33,975]
        var_targets=[ [1,1], None]
        eval_names =  ['MetroU0_e1t31_vartarget_eval', 
                       'NWB_VarialeEscapeInit_eval' ]
        eval_nums = [1000,200]
        evalResults={}

        for world_name, node_maxim, var_target, eval_name, eval_num in zip(world_list, node_maxims, var_targets, eval_names, eval_nums):
        #custom_env = CreateEnv('MetroU3_e1t31_FixedEscapeInit',max_nodes=33,nfm_func_name =config['nfm_func_name'],var_targets=[1,1], remove_world_pool=True, apply_wrappers=True)
        #evalName='MetroU0_e1t31_vartarget_eval'
        
            custom_env = CreateEnv(world_name, max_nodes=node_maxim, nfm_func_name = config['nfm_func_name'], var_targets=var_target, remove_world_pool=True, apply_wrappers=True)
            evalName=eval_name
            senv=SuperEnv([custom_env],{1:0},node_maxim,probs=[1])        
            n_eval=eval_num
            evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
            for seed in range(SEED0, SEED0+NUMSEEDS):
                saved_policy = s2v_ActorCriticPolicy.load(LOGDIR+'/SEED'+str(seed)+"/saved_models/policy_last")
                saved_policy_deployable=DeployablePPOPolicy(custom_env, saved_policy)
                ppo_policy = ActionMaskedPolicySB3_PPO(saved_policy_deployable, deterministic=True)
                #result = evaluate_ppo(logdir=config['logdir']+'/SEED'+str(seed), policy=ppo_policy, config=config, env=[custom_env], eval_subdir=evalName)
                result = evaluate_ppo(logdir=config['logdir']+'/SEED'+str(seed), policy=ppo_policy, config=config, env=senv, eval_subdir=evalName, n_eval=n_eval)
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

