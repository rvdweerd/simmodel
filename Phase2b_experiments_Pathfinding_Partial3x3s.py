import copy
from modules.gnn.comb_opt import train, evaluate, evaluate_spath_heuristic, evaluate_tabular
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us
from modules.sim.graph_factory import GetWorldSet, LoadData
from modules.sim.simdata_utils import SimulateInteractiveMode
import numpy as np
import torch
import argparse
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from Phase2_generate_partial_graphs import print_world_properties

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    # Model hyperparameters
    parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('--emb_itT', default=2, type=int)
    parser.add_argument('--num_epi', default=250, type=int)
    parser.add_argument('--mem_size', default=2000, type=int)
    parser.add_argument('--nfm_func', default='NFM_ev_ec_t', type=str)
    parser.add_argument('--scenario', default='None', type=str)
    parser.add_argument('--optim_target', default='None', type=str)
    parser.add_argument('--tau', default=100, type=int)
    parser.add_argument('--nstep', default=1, type=int)
    parser.add_argument('--train', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--eval', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])   
    parser.add_argument('--Etrain', default=[], type=list)
    parser.add_argument('--Utrain', default=[], type=list)
    parser.add_argument('--num_seeds', default=5, type=int)
    parser.add_argument('--seed0', default=10, type=int)
    parser.add_argument('--solve_select', default='solvable', type=str)
    parser.add_argument('--edge_blocking', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    
    args=parser.parse_args()

    scenario_name=args.scenario
    #scenario_name = 'Train_U2E45'
    world_name = 'SubGraphsManhattan3x3'
    state_repr = 'etUte0U0'
    state_enc  = 'nfm'
    nfm_funcs = {'NFM_ev_ec_t':NFM_ev_ec_t(),'NFM_ec_t':NFM_ec_t(),'NFM_ev_t':NFM_ev_t(),'NFM_ev_ec_t_um_us':NFM_ev_ec_t_um_us()}
    nfm_func=nfm_funcs[args.nfm_func]
    edge_blocking = args.edge_blocking
    solve_select = args.solve_select # only solvable worlds (so best achievable performance is 100%)
    reject_u_duplicates = False
    Etrain=args.Etrain
    Etrain=[int(i) for i in Etrain if i.isnumeric() ]#[4,5]
    Utrain=args.Utrain#[2]
    Utrain=[int(i) for i in Utrain if i.isnumeric() ]

    databank_full, register_full, solvable = LoadData(edge_blocking = True)
    env_all_train, hashint2env, env2hashint, env2hashstr = GetWorldSet(state_repr, state_enc, U=Utrain, E=Etrain, edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_u_duplicates, nfm_func=nfm_func)
    
    # env_idx=0
    # env=env_all_train[env_idx]
    # entry=env.current_entry
    # hashint=env2hashint[env_idx]
    # hashstr=env2hashstr[env_idx]
    # u=env.sp.U
    # s= solvable['U='+str(u)][hashint]
    # print_world_properties(env, env_idx, entry, hashint, hashstr, edge_blocking, solve_select, reject_u_duplicates, solvable_=s)    
    # SimulateInteractiveMode(env_all_train[0])

    config={}
    config['node_dim']      = env_all_train[0].F
    config['max_num_nodes']  = 9#env_all_train[0].sp.V
    config['scenario_name'] = args.scenario
    config['nfm_func']      = args.nfm_func
    config['emb_dim']       = args.emb_dim
    config['emb_iter_T']    = args.emb_itT 
    config['optim_target']  = args.optim_target
    #config['num_extra_layers']=0        
    config['num_episodes']  = args.num_epi 
    config['memory_size']   = args.mem_size 
    config['num_step_ql']   = args.nstep  
    config['bsize']         = 32        
    config['gamma']         = .9        
    config['lr_init']       = 1e-3      
    config['lr_decay']      = 0.99999    
    config['tau']           = args.tau  # num grad steps for each target network update
    config['eps_0']         = 1.        
    config['eps_min']       = 0.1
    epi_min                 = .9        # reach eps_min at % of episodes # .9
    config['eps_decay']     = 1 - np.exp(np.log(config['eps_min'])/(epi_min*config['num_episodes']))
    rootdir='./results_Phase2/Pathfinding/'+ \
                                world_name+'/'+ \
                                solve_select+'_edgeblock'+str(edge_blocking)+'/'+\
                                scenario_name
    config['logdir']        = rootdir + '/' +\
                                nfm_func.name+'/'+ \
                                'emb'+str(config['emb_dim']) + \
                                '_itT'+str(config['emb_iter_T']) + \
                                '_epi'+str(config['num_episodes']) + \
                                '_mem'+str(config['memory_size']) + \
                                '_nstep'+str(config['num_step_ql'])
    numseeds=args.num_seeds
    seed0=args.seed0


    #
    #   Train the model on selected subset of graphs
    #
    if args.train:
        for seed in range(seed0, seed0+numseeds):
            train(seed=seed, config=config, env_all=env_all_train)

    #
    #   Evaluation
    #
    # Evaluate with simple shortest path heuristic to get low mark on performance 
    if args.eval:
        evalResults={}
        #evaluate_spath_heuristic(logdir=rootdir+'/heur/spath', config=config, env_all=env_all_train)
        # Evaluate on the full training set
        evalName='trainset_eval'
        evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
        for seed in range(seed0, seed0+numseeds):
            result = evaluate(logdir=config['logdir']+'/SEED'+str(seed), config=config, env_all=env_all_train, eval_subdir=evalName)
            num_unique_graphs, num_graph_instances, avg_return, success_rate = result
            evalResults[evalName]['num_graphs.........'].append(num_unique_graphs)
            evalResults[evalName]['num_graph_instances'].append(num_graph_instances)
            evalResults[evalName]['avg_return.........'].append(avg_return)
            evalResults[evalName]['success_rate.......'].append(success_rate)

        # Evaluate on the full evaluation set
        evalName='testset_eval'
        evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
        Etest=[0,1,2,3,4,5,6,7,8,9,10]
        Utest=[1,2,3]
        env_all_test, _, _, _  = GetWorldSet(state_repr, state_enc, U=Utest, E=Etest, edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_u_duplicates, nfm_func=nfm_func)
        for seed in range(seed0, seed0+numseeds):
            result = evaluate(logdir=config['logdir']+'/SEED'+str(seed), config=config, env_all=env_all_test, eval_subdir=evalName)
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
                    env_all_test, _, _, _  = GetWorldSet(state_repr, state_enc, U=[u], E=[e], edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_u_duplicates, nfm_func=nfm_func)
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
            printing('\nWeighted return: '+weighted_return)
            printing('Weighted success rate: '+weighted_success_rate)
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
            for seed in range(seed0, seed0+numseeds):
                evalName=world_name[:16]+'_eval'
                evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
                custom_env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
                custom_env.redefine_nfm(nfm_func)
                result = evaluate(logdir=config['logdir']+'/SEED'+str(seed0), config=config, env_all=[custom_env], eval_subdir=evalName)
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

