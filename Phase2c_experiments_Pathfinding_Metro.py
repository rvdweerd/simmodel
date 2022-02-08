import copy
from modules.gnn.comb_opt import train, evaluate, evaluate_spath_heuristic, evaluate_tabular
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us, NFM_ev_ec_t_u
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
    parser.add_argument('--optim_target', default='None', type=str)
    parser.add_argument('--tau', default=100, type=int)
    parser.add_argument('--nstep', default=1, type=int)
    parser.add_argument('--edge_blocking', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--train', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--eval', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])   
    parser.add_argument('--num_seeds', default=5, type=int)
    parser.add_argument('--seed0', default=10, type=int)
    args=parser.parse_args()

    nfm_funcs = {
        'NFM_ev_ec_t'       : NFM_ev_ec_t(),
        'NFM_ec_t'          : NFM_ec_t(),
        'NFM_ev_t'          : NFM_ev_t(),
        'NFM_ev_ec_t_u'     : NFM_ev_ec_t_u(),
        'NFM_ev_ec_t_um_us' : NFM_ev_ec_t_um_us(),
    }
    nfm_func=nfm_funcs[args.nfm_func]
    edge_blocking = args.edge_blocking
    solve_select = 'solvable' # only solvable worlds (so best achievable performance is 100%)

    world_name='MetroU3_e17tborder_FixedEscapeInit'
    scenario_name='TrainMetro'
    state_repr='etUte0U0'
    state_enc='nfm'
    
    custom_env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    custom_env.redefine_nfm(nfm_func)
    custom_env.capture_on_edges = edge_blocking
    env_all_train=[ custom_env ]

    config = {}
    numseeds = args.num_seeds
    seed0 = args.seed0
    config['node_dim']      = env_all_train[0].F
    config['max_num_nodes'] = env_all_train[0].sp.V
    config['scenario_name'] = scenario_name
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
                                scenario_name
    config['logdir']        = rootdir + '/' +\
                                'eblock'+str(edge_blocking) +'/'+\
                                nfm_func.name+'/'+ \
                                'emb'+str(config['emb_dim']) + \
                                '_itT'+str(config['emb_iter_T']) + \
                                '_epi'+str(config['num_episodes']) + \
                                '_mem'+str(config['memory_size']) + \
                                '_nstep'+str(config['num_step_ql']) \
                                 

    # Evaluate with simple shortest path heuristic to get low mark on performance 
    #evaluate_spath_heuristic(logdir=rootdir+'/ebblock'+str(edge_blocking)+'heur/spath', config=config, env_all=env_all_train)

    #
    #   Train the model on selected subset of graphs
    #
    if args.train:
        for seed in range(seed0, seed0+numseeds):
            train(seed=seed, config=config, env_all=env_all_train)

    #
    #   Evaluation
    #
    # Evaluate on the full training set
    if args.eval:
        evalResults={}

        world_names=[
            'Manhattan5x5_FixedEscapeInit',
            'Manhattan5x5_VariableEscapeInit',
            'MetroU3_e17tborder_FixedEscapeInit',
            'SparseManhattan5x5',
            'MetroU3_e1t31_FixedEscapeInit'
            'MetroU3_e17t31_FixedEscapeInit', 
            'MetroU3_e17t0_FixedEscapeInit',
        ]
        state_repr='etUte0U0'
        state_enc='nfm'
        for world_name in world_names:
            evalName=world_name[:16]+'_eval'
            evalResults[evalName]={'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} 
            custom_env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
            custom_env.redefine_nfm(nfm_func)
            custom_env.capture_on_edges = edge_blocking

            for seed in range(seed0, seed0+numseeds):
                result = evaluate(logdir=config['logdir']+'/SEED'+str(seed), config=config, env_all=[custom_env], eval_subdir=evalName)
                num_unique_graphs, num_graph_instances, avg_return, success_rate = result
                evalResults[evalName]['num_graphs.........'].append(num_unique_graphs)
                evalResults[evalName]['num_graph_instances'].append(num_graph_instances)
                evalResults[evalName]['avg_return.........'].append(avg_return)
                evalResults[evalName]['success_rate.......'].append(success_rate)


        assert False
        env_all_test=env_all_train
        evalName='trainseteval'
        evalResults={ evalName:{'num_graphs.........':[],'num_graph_instances':[],'avg_return.........':[],'success_rate.......':[],} }
        for seed in range(seed0, seed0+numseeds):
            result = evaluate(logdir=config['logdir']+'/SEED'+str(seed), config=config, env_all=env_all_test, eval_subdir=evalName)
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
            