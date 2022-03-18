import copy
from modules.gnn.comb_opt import train, evaluate, evaluate_spath_heuristic, evaluate_tabular
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.environments import SuperEnv
#from modules.gnn.nfm_gen import NFM_ec_t, NFM_ec_t_dt_at, NFM_ev_ec_t_dt_at_um_us, NFM_ec_dt, NFM_ec_dtscaled, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us
import modules.gnn.nfm_gen
from modules.sim.graph_factory import GetWorldSet, LoadData
from modules.sim.simdata_utils import SimulateInteractiveMode, SimulateAutomaticMode_DQN
from modules.gnn.comb_opt import init_model
from modules.rl.rl_policy import GNN_s2v_Policy
from modules.ppo.helpfuncs import CreateEnv
import numpy as np
import random
import torch
import argparse
from Phase2d_construct_sets import ConstructTrainSet
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def GetConfig(args):
    config={}
    config['train_on'] = args.train_on
    config['max_nodes'] = args.max_nodes
    config['remove_paths'] = args.pursuit == 'Uoff'
    assert args.pursuit in ['Uoff','Uon']
    #state_repr = 'etUte0U0'
    #state_enc = 'nfm'
    config['reject_u_duplicates'] = False
    config['Etrain'] = [int(i) for i in args.Etrain if i.isnumeric() ]
    config['Utrain'] = [int(i) for i in args.Utrain if i.isnumeric() ]
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
    config['num_episodes'] = args.num_epi 
    config['memory_size']  = args.mem_size 
    config['num_step_ql']  = args.nstep  
    config['bsize']        = 32        
    config['gamma']        = .9        
    config['lr_init']      = 1e-3      
    config['lr_decay']     = 0.99999    
    config['tau']          = args.tau  # num grad steps for each target network update
    config['eps_0']        = 1.        
    config['eps_min']      = 0.1
    config['epi_min']      = .9        # reach eps_min at % of episodes # .9
    config['eps_decay']    = 1 - np.exp(np.log(config['eps_min'])/(config['epi_min']*config['num_episodes']))
    config['rootdir']='./results/results_Phase2/Pathfinding/dqn/'+ \
                                config['train_on']+'_'+args.pursuit+'/'+ \
                                config['solve_select']+'_edgeblock'+str(config['edge_blocking'])+'/'+\
                                config['qnet']+'_normagg'+str(config['norm_agg'])
    config['logdir']        = config['rootdir'] + '/' +\
                                config['nfm_func']+'/'+ \
                                'emb'+str(config['emb_dim']) + \
                                '_itT'+str(config['emb_iter_T']) + \
                                '_epi'+str(config['num_episodes']) + \
                                '_mem'+str(config['memory_size']) + \
                                '_nstep'+str(config['num_step_ql'])
    config['seed0']=args.seed0
    config['numseeds']=args.num_seeds
    config['seedrange']=range(config['seed0'], config['seed0']+config['numseeds'])
    print('seedrange')
    for i in config['seedrange']: print(i)
    return config

def main(args):
    config=GetConfig(args)
    print('device',device)
    #
    #   Load and test trainset
    #
    #databank_full, register_full, solvable = LoadData(edge_blocking = True)
    #env_all_train, hashint2env, env2hashint, env2hashstr = GetWorldSet(state_repr, state_enc, U=Utrain, E=Etrain, edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_u_duplicates, nfm_func=nfm_func)
    if args.train or args.eval:
        senv, env_all_train_list = ConstructTrainSet(config, apply_wrappers=False, remove_paths=config['remove_paths'], tset=config['train_on']) #TODO check
        env_all_train = [senv]
        if config['demoruns']:
            while True:
                a = SimulateInteractiveMode(senv, filesave_with_time_suffix=False)
                if a == 'Q': break

    #
    #   Train the model on selected subset of graphs
    #
    if args.train:
        for seed in config['seedrange']:
            train(seed=seed, config=config, env_all=env_all_train)

    #
    #   Evaluation
    #
    if args.eval:
        evalResults={}
        test_heuristics              = True
        test_full_trainset           = False
        test_full_solvable_3x3subs   = False
        test_all_solvable_3x3segments= False
        test_other_worlds            = False
        
        if test_heuristics:
            # Evaluate with simple shortest path heuristic on full trainet to get low mark on performance 
            evaluate_spath_heuristic(logdir=config['rootdir']+'/heur/spath', config=config, env_all=env_all_train_list)

        if test_full_trainset:
            # Evaluate on the full training set
            evalName='trainset_eval'
            evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
            for seed in config['seedrange']:
                result = evaluate(logdir=config['logdir']+'/SEED'+str(seed), config=config, env_all=env_all_train_list, eval_subdir=evalName)
                num_unique_graphs, num_graph_instances, avg_return, success_rate = result
                evalResults[evalName]['num_graphs.........'].append(num_unique_graphs)
                evalResults[evalName]['num_graph_instances'].append(num_graph_instances)
                evalResults[evalName]['avg_return.........'].append(avg_return)
                evalResults[evalName]['success_rate.......'].append(success_rate)

        if test_full_solvable_3x3subs:
            # Evaluate on the full evaluation set
            evalName='testset_eval'
            evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
            Etest=[0,1,2,3,4,5,6,7,8,9,10]
            Utest=[1,2,3]
            env_all_test, _, _, _  = GetWorldSet(state_repr, state_enc, U=Utest, E=Etest, edge_blocking=config['edge_blocking'], solve_select=config['solve_select'], reject_duplicates=config['reject_u_duplicates'], nfm_func=modules.gnn.nfm_gen.nfm_funcs[config['nfm_func']])
            for seed in config['seedrange']:
                result = evaluate(logdir=config['logdir']+'/SEED'+str(seed), config=config, env_all=env_all_test, eval_subdir=evalName)
                num_unique_graphs, num_graph_instances, avg_return, success_rate = result
                evalResults[evalName]['num_graphs.........'].append(num_unique_graphs)
                evalResults[evalName]['num_graph_instances'].append(num_graph_instances)
                evalResults[evalName]['avg_return.........'].append(avg_return)
                evalResults[evalName]['success_rate.......'].append(success_rate)
        if test_all_solvable_3x3segments:
            # Evaluate on each individual segment of the evaluation set
            evalName='graphsegments_eval'
            evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
            for seed in config['seedrange']:
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
                        env_all_test, _, _, _  = GetWorldSet(state_repr, state_enc, U=[u], E=[e], edge_blocking=config['edge_blocking'], solve_select=config['solve_select'], reject_duplicates=config['reject_u_duplicates'], nfm_func=modules.gnn.nfm_gen.nfm_funcs[config['nfm_func']])
                        result = evaluate(logdir=config['logdir']+'/SEED'+str(seed), config=config, env_all=env_all_test, eval_subdir=evalName+'/runs/'+'E'+str(e)+'U'+str(u))
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
        if test_other_worlds:
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
                custom_env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
                custom_env.redefine_nfm(modules.gnn.nfm_gen.nfm_funcs[config['nfm_func']])
                for seed in range(config['seed0'], config['seed0']+config['numseeds']):
                    result = evaluate(logdir=config['logdir']+'/SEED'+str(seed), config=config, env_all=[custom_env], eval_subdir=evalName)
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

    #
    #   Test on unseen graphs
    #           
    if args.test:
        evalResults={}
        world_list=[
            # 'Manhattan5x5_DuplicateSetB',
            # 'Manhattan3x3_WalkAround',
            # 'MetroU3_e1t31_FixedEscapeInit', 
            # 'full_solvable_3x3subs',
            'Manhattan5x5_FixedEscapeInit',
            'Manhattan5x5_VariableEscapeInit',
            'MetroU3_e17tborder_FixedEscapeInit',
            'MetroU3_e17tborder_VariableEscapeInit',
            'NWB_ROT_FixedEscapeInit',
            'NWB_ROT_VariableEscapeInit',
            'NWB_test_FixedEscapeInit',
            'NWB_test_VariableEscapeInit',
            'NWB_UTR_FixedEscapeInit',
            'NWB_UTR_VariableEscapeInit',            
            'SparseManhattan5x5',
            ]
        #node_maxims = [0,0,0,0]
        #var_targets=[ None,None,None,None]
        #eval_names = world_list
        #eval_nums = [10,10,10,10]

        for world_name in world_list:
            evalName=world_name
            if world_name == 'full_solvable_3x3subs':
                Etest=[0,1,2,3,4,5,6,7,8,9,10]
                Utest=[1,2,3]
                evalenv, _, _, _  = GetWorldSet('etUte0U0', 'nfm', U=Utest, E=Etest, edge_blocking=config['edge_blocking'], solve_select=config['solve_select'], reject_duplicates=False, nfm_func=modules.gnn.nfm_gen.nfm_funcs[config['nfm_func']])
            else:
                env = CreateEnv(world_name, max_nodes=0, nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=config['remove_paths'], apply_wrappers=False)
                evalenv=[env]
            #env = CreateEnv('NWB_test_FixedEscapeInit',max_nodes=975,nfm_func_name = config['nfm_func'],var_targets=None, remove_world_pool=True, apply_wrappers=False)
            #env = CreateEnv('MetroU3_e17tborder_FixedEscapeInit',max_nodes=33,nfm_func_name = config['nfm_func'],var_targets=[1,1], remove_world_pool=True, apply_wrappers=False)
            #env = CreateEnv('MetroU3_e1t31_FixedEscapeInit',max_nodes=33,nfm_func_name = config['nfm_func'],var_targets=[1,1], remove_world_pool=True, apply_wrappers=False)        
            #env = CreateEnv('MetroU3_e17tborder_VariableEscapeInit',max_nodes=33,nfm_func_name = config['nfm_func'],var_targets=None, remove_world_pool=True, apply_wrappers=False)        
            #env, env_all_train_list = ConstructTrainSet(config, apply_wrappers=False, remove_paths=config['remove_paths'], tset=config['train_on']) #TODO check

            # calcHeur=True
            # if calcHeur:
            #     evaluate_spath_heuristic(logdir=config['rootdir']+'/heur/'+evalName, config=config, env_all=evalenv)
            # continue
            if config['demoruns']:
                Q_func, Q_net, optimizer, lr_scheduler = init_model(config,fname=config['logdir']+'/SEED'+str(config['seed0'])+'/best_model.tar')
                policy=GNN_s2v_Policy(Q_func)
                while True:
                    entries=None#[5012,218,3903]
                    env = random.choice(evalenv)
                    a = SimulateAutomaticMode_DQN(env, policy, t_suffix=False, entries=entries)
                    if a == 'Q': break
            
            #evalenv=SuperEnv([env], {1:0}, node_maxim, probs=[1])
            #evalenv=[env]
            #evalName='MetroU0_e1t31_vartarget_eval'
            #evalName=eval_name
            #n_eval=eval_num
            evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
            for seed in config['seedrange']:
                logdir=config['logdir']+'/SEED'+str(seed)
                result = evaluate(logdir=config['logdir']+'/SEED'+str(seed), config=config, env_all=evalenv, eval_subdir=evalName)
                #result = evaluate(logdir=config['logdir']+'/SEED'+str(seed), config=config, env_all=[env], eval_subdir=evalName)
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
    # Model hyperparameters
    parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('--emb_itT', default=2, type=int)
    parser.add_argument('--num_epi', default=250, type=int)
    parser.add_argument('--mem_size', default=2000, type=int)
    parser.add_argument('--nfm_func', default='NFM_ev_ec_t', type=str)
    parser.add_argument('--train_on', default='None', type=str)
    parser.add_argument('--qnet', default='None', type=str)
    parser.add_argument('--norm_agg', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--optim_target', default='None', type=str)
    parser.add_argument('--tau', default=100, type=int)
    parser.add_argument('--nstep', default=1, type=int)
    parser.add_argument('--train', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--eval', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])   
    parser.add_argument('--test', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])       
    parser.add_argument('--Etrain', default=[], type=list)
    parser.add_argument('--Utrain', default=[], type=list)
    parser.add_argument('--num_seeds', default=5, type=int)
    parser.add_argument('--seed0', default=10, type=int)
    parser.add_argument('--solve_select', default='solvable', type=str)
    parser.add_argument('--edge_blocking', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--max_nodes', default=9, type=int)
    parser.add_argument('--demoruns', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--pursuit', default='Uon', type=str)
    args=parser.parse_args()
    main(args)