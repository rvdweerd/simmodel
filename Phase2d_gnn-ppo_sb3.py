import argparse
#import gym
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
#from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from modules.gnn.comb_opt import evaluate_spath_heuristic
from modules.rl.rl_utils import EvaluatePolicy, print_parameters, GetFullCoverageSample, NpWrapper
#from modules.sim.simdata_utils import SimulateInteractiveMode
#import modules.sim.simdata_utils as su
from modules.ppo.callbacks_sb3 import SimpleCallback, TestCallBack
from modules.ppo.models_sb3 import s2v_ActorCriticPolicy, Struc2Vec
#from typing import Callable, Dict, List, Optional, Tuple, Type, Union
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from modules.ppo.helpfuncs import get_super_env, CreateEnv, eval_simple, evaluate_ppo
from Phase2d_construct_sets import ConstructTrainSet
if __name__ == '__main__':
    train           =True
    eval            =False
    #train_on        ='SubGraphsManhattan3x3'
    #train_on        = 'Manhattan5x5_VariableEscapeInit'
    train_on        = 'ContructedSuperSet'
    solve_select    ='both'
    edge_blocking   =True
    Utrain          =[1,2,3]
    Etrain          =[0,1,2,3,4,5,6,7,8,9]
    ustr=''
    for i in Utrain: ustr+=str(i)
    estr=''
    for i in Etrain: estr+=str(i)
    #scenario_name   = 'Train_U'+ustr+'E'+estr
    scenario_name   = ''
    nfm_func_name   = 'NFM_ev_ec_t_um_us'
    emb_dim         = 64
    emb_iter_T      =  5
    num_step        = 500000#300000
    seed0           = 0
    numseeds        = 1
    max_nodes       = 33
    config={}
    config['train_on'] = train_on
    config['solve_select'] = solve_select
    config['edge_blocking'] = edge_blocking
    config['Utrain'] = Utrain
    config['Etrain'] = Etrain
    config['nfm_func_name'] = nfm_func_name
    config['emb_dim'] = emb_dim
    config['emb_iter_T'] = emb_iter_T
    config['num_step'] = num_step
    config['seed0'] = seed0
    config['numseeds'] = numseeds
    config['max_nodes'] = max_nodes

    rootdir='results/results_Phase2/Pathfinding/ppo/'+train_on+'/'+solve_select+'_edgeblock'+str(edge_blocking)+'/'+scenario_name
    logdir=rootdir+'/'+\
            nfm_func_name+'/'+ \
            'emb'+str(emb_dim) + \
            '_itT'+str(emb_iter_T) + \
            '_nstep'+str(num_step)
    config['rootdir']=rootdir
    config['logdir']=logdir

    if train:
        MAX_NODES=max_nodes
        EMB_DIM = emb_dim
        EMB_ITER_T = emb_iter_T
        TOTAL_TIME_STEPS = num_step
        #env=CreateEnv(world_name='Manhattan3x3_WalkAround', max_nodes=MAX_NODES)
        #env=CreateEnv(world_name=config['train_on'], max_nodes=MAX_NODES)
        #env, _ = get_super_env(Uselected=Utrain, Eselected=Etrain, config=config)
        env = ConstructTrainSet(config)

        obs=env.reset()
        NODE_DIM = env.F

        policy_kwargs = dict(
            features_extractor_class=Struc2Vec,
            features_extractor_kwargs=dict(emb_dim=EMB_DIM, emb_iter_T=EMB_ITER_T, node_dim=NODE_DIM),#, num_nodes=MAX_NODES),
            # NOTE: FOR THIS TO WORK, NEED TO ADJUST sb3 policies.py
            #           def proba_distribution_net(self, latent_dim: int) -> nn.Module:
            #       to create a linear layer that maps to 1-dim instead of self.action_dim
            #       reason: our model is invariant to the action space (number of nodes in the graph) 
        )

        for seed in range(seed0, seed0+numseeds):
            logdir_ = logdir+'/SEED'+str(seed)

            # eval_callback = EvalCallback(
            #     env_eval,
            #     best_model_save_path=logdir_+'./evalcallbacklogs/',
            #     log_path=logdir_+'/evalcallbacklogs/', eval_freq=1000,
            #     deterministic=True,
            #     render=False)
            # run = wandb.init(
            #     project="sb3",
            #     config={'config1':1, 'config2':2},
            #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            #     # monitor_gym=True,  # auto-upload the videos of agents playing the game
            #     # save_code=True,  # optional
            # )
            # wandb_callback=WandbCallback(
            #     model_save_path=f"models/{run.id}",
            #     verbose=2,
            #     gradient_save_freq=100,
            # )

            model = MaskablePPO(s2v_ActorCriticPolicy, env, \
                #learning_rate=1e-4,\
                seed=seed,\
                #clip_range=0.1,\    
                #max_grad_norm=0.1,\
                policy_kwargs = policy_kwargs, verbose=2, tensorboard_log=logdir_+"/tb/")

            print_parameters(model.policy)
            model.learn(total_timesteps = TOTAL_TIME_STEPS, callback=[TestCallBack()])#,eval_callback]) #,wandb_callback])
            # run.finish()
            model.save(logdir_+"/saved_models/model_last")
            policy = model.policy
            policy.save(logdir_+"/saved_models/policy_last")    

    if eval:
        # nfm_funcs = {
        #     'NFM_ev_ec_t'       : NFM_ev_ec_t(),
        #     'NFM_ec_t'          : NFM_ec_t(),
        #     'NFM_ev_t'          : NFM_ev_t(),
        #     'NFM_ev_ec_t_u'     : NFM_ev_ec_t_u(),
        #     'NFM_ev_ec_t_um_us' : NFM_ev_ec_t_um_us(),
        # }
        # nfm_func=nfm_funcs[args.nfm_func]
        # edge_blocking = args.edge_blocking
        # solve_select = 'solvable' # only solvable worlds (so best achievable performance is 100%)

        # world_name='MetroU3_e17tborder_FixedEscapeInit'
        # scenario_name='TrainMetro'
        # state_repr='etUte0U0'
        # state_enc='nfm'
        #MAX_NODES=max_nodes
        #EMB_DIM = emb_dim
        #EMB_ITER_T = emb_iter_T

        #env=CreateEnv(world_name='Manhattan3x3_WalkAround', max_nodes=MAX_NODES)
        #env=get_super_env(Utrain=[1], Etrain=[4],max_nodes=MAX_NODES)
        #env=CreateEnv(world_name='MetroU3_e17tborder_FixedEscapeInit', max_nodes=MAX_NODES)
        #env=CreateEnv(world_name='Manhattan3x3_WalkAround', max_nodes=MAX_NODES)
        #SimulateInteractiveMode(env,filesave_with_time_suffix=False)
        #NODE_DIM = env.F
        #model=PPO.load('ppo_trained_on_all_3x3')
        #check_custom_position_probs(env,saved_policy,hashint=4041,entry=5,targetnodes=[1],  epath=[4],upaths=[[6]], max_nodes=MAX_NODES)
        #res = evaluate_policy(saved_policy, env, n_eval_episodes=20, reward_threshold=-100, warn=False, return_episode_rewards=True)
        #print('Test result: avg rew:', res[0], 'std:', res[1])
        #eval_simple(saved_policy,env)

        #_, env_all_train = get_super_env(Uselected=Utrain, Eselected=Etrain, config=config)
        _, env_all_train = get_super_env(Uselected=[1], Eselected=[0], config=config)

        evalResults={}
        #evaluate_spath_heuristic(logdir=rootdir+'/heur/spath', config=config, env_all=env_all_train)
        # Evaluate on the full training set
        evalName='trainset_eval'
        evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 


        for seed in range(seed0, seed0+numseeds):
            #logdir_ = logdir+'/SEED'+str(seed)
            saved_policy = s2v_ActorCriticPolicy.load(logdir+'/SEED'+str(seed)+"/saved_models/policy_last")

            #CREATE POLICY CLASS

            result = evaluate_ppo(logdir=logdir+'/SEED'+str(seed), policy=..., config=config, env_all=env_all_train, eval_subdir=evalName)
            num_unique_graphs, num_graph_instances, avg_return, success_rate = result
            evalResults[evalName]['num_graphs.........'].append(num_unique_graphs)
            evalResults[evalName]['num_graph_instances'].append(num_graph_instances)
            evalResults[evalName]['avg_return.........'].append(avg_return)
            evalResults[evalName]['success_rate.......'].append(success_rate)

        # Evaluate on the full evaluation set
        evalName='testset_eval'
        evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
        Utest=[0,1,2,3]
        Etest=[0,1,2,3,4,5,6,7,8,9,10]
        _, env_all_test = get_super_env(Uselected=Utest, Eselected=Etest, config=config)
        for seed in range(seed0, seed0+numseeds):
            result = evaluate_ppo(logdir=config['logdir']+'/SEED'+str(seed), config=config, env_all=env_all_test, eval_subdir=evalName)
            num_unique_graphs, num_graph_instances, avg_return, success_rate = result
            evalResults[evalName]['num_graphs.........'].append(num_unique_graphs)
            evalResults[evalName]['num_graph_instances'].append(num_graph_instances)
            evalResults[evalName]['avg_return.........'].append(avg_return)
            evalResults[evalName]['success_rate.......'].append(success_rate)
        
        # Evaluate on each individual segment of the evaluation set
        evalName='graphsegments_eval'
        evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
        for seed in range(seed0, seed0+numseeds):
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
                    result = evaluate_ppo(logdir=config['logdir']+'/SEED'+str(seed), config=config, env_all=env_all_test, eval_subdir=evalName+'/runs/'+'E'+str(e)+'U'+str(u))
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
        world_names=[
            'Manhattan5x5_FixedEscapeInit',
            'Manhattan5x5_VariableEscapeInit',
            'MetroU3_e17tborder_FixedEscapeInit',
            'SparseManhattan5x5',
        ]
        state_repr='etUte0U0'
        state_enc='nfm'
        for world_name in world_names:
            evalName=world_name[:16]+'_eval'
            evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
            
            custom_env=CreateEnv(world_name=world_name, max_nodes=config['max_nodes'])
            #custom_env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
            #custom_env.redefine_nfm(nfm_func)
            for seed in range(seed0, seed0+numseeds):
                result = evaluate_ppo(logdir=config['logdir']+'/SEED'+str(seed), config=config, env_all=[custom_env], eval_subdir=evalName)
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