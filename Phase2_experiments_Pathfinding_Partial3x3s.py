import copy
from modules.gnn.comb_opt import train, evaluate
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
    args=parser.parse_args()

    #scenario_name=args.scenario
    scenario_name = 'Train_U2E45_TestU2E36'
    world_name = 'SubGraphsManhattan3x3'
    state_repr = 'etUte0U0'
    state_enc  = 'nfm'
    nfm_funcs = {'NFM_ev_ec_t':NFM_ev_ec_t(),'NFM_ec_t':NFM_ec_t(),'NFM_ev_t':NFM_ev_t(),'NFM_ev_ec_t_um_us':NFM_ev_ec_t_um_us()}
    nfm_func=nfm_funcs[args.nfm_func]
    
    Utrain=[2]
    Etrain=[5]
    Utest=[2]
    Etest=[3,6]
    edge_blocking = True
    solve_select = 'solvable' # only solvable worlds (so best achievable performance is 100%)
    reject_u_duplicates = False
    
    databank_full, register_full, solvable = LoadData(edge_blocking = True)
    env_all_train, hashint2env, env2hashint, env2hashstr = GetWorldSet(state_repr, state_enc, U=Utrain, E=Etrain, edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_u_duplicates, nfm_func=nfm_func)
    env_all_test, _, _, _  = GetWorldSet(state_repr, state_enc, U=Utest, E=Etest, edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_u_duplicates, nfm_func=nfm_func)
    env_idx=0
    env=env_all_train[env_idx]
    entry=env.current_entry
    hashint=env2hashint[env_idx]
    hashstr=env2hashstr[env_idx]
    u=env.sp.U
    s= solvable['U='+str(u)][hashint]
    print_world_properties(env, env_idx, entry, hashint, hashstr, edge_blocking, solve_select, reject_u_duplicates, solvable_=s)    
    SimulateInteractiveMode(env_all_train[0])
    
    config={}
    config['node_dim']      = env_all_train[0].F
    config['num_nodes']     = env_all_train[0].sp.V
    config['scenario_name'] = args.scenario
    config['nfm_func']      = args.nfm_func
    config['emb_dim']       = args.emb_dim
    config['emb_iter_T']    = args.emb_itT 
    config['optim_target']  = args.optim_target
    #config['num_extra_layers']=0        
    config['num_episodes']  = args.num_epi 
    config['memory_size']   = args.mem_size 
    config['num_step_ql']   = args.nstep  
    config['bsize']         = 64        
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
    config['logdir']        = rootdir + '/TEST/' + \
                                nfm_func.name+'/'+ \
                                '_emb'+str(config['emb_dim']) + \
                                '_itT'+str(config['emb_iter_T']) + \
                                '_epi'+str(config['num_episodes']) + \
                                '_mem'+str(config['memory_size']) + \
                                '_nstep'+str(config['num_step_ql'])
    numseeds=1
    seed0=0
    train(seeds=numseeds, seednr0=seed0, config=config, env_all=env_all_train)
    evaluate(logdir=config['logdir']+'/SEED'+str(seed0), config=config, env_all=env_all_train, eval_subdir='traineval')
    evaluate(logdir=config['logdir']+'/SEED'+str(seed0), config=config, env_all=env_all_test, eval_subdir='testeval' )
    #evaluate_spath_heuristic(logdir=rootdir+'/heur/spath', config, env_all=env_all)
    #evaluate_tabular(logdir=rootdir+'/tabular', config, env_all=env_all)