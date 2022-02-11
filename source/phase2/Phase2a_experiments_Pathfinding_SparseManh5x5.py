import copy
from modules.gnn.comb_opt import train, evaluate
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us
import numpy as np
import torch
import argparse
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    scenario_name=args.scenario
    nfm_funcs = {'NFM_ev_ec_t':NFM_ev_ec_t(),'NFM_ec_t':NFM_ec_t(),'NFM_ev_t':NFM_ev_t(),'NFM_ev_ec_t_um_us':NFM_ev_ec_t_um_us()}
    world_name='SparseManhattan5x5'
    state_repr='etUt'
    state_enc='nfm'

    env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    nfm_func = nfm_funcs[args.nfm_func]
    env.redefine_nfm(nfm_func)
    env_all=[]
    if scenario_name=='1target-fixed24':
        env._remove_world_pool()
        env.databank={}
        env.register={}
        env.redefine_goal_nodes([24])
        env.current_entry=1
        env_all.append(copy.deepcopy(env))
    elif scenario_name=='1target-random':
        env._remove_world_pool()
        env.databank={}
        env.register={}
        for i in range(env.sp.V):
            if i == env.sp.coord2labels[env.sp.start_escape_route]: 
                continue
            env.redefine_goal_nodes([i])
            env.current_entry=i
            env_all.append(copy.deepcopy(env))
    elif scenario_name=='2target-random':
        env._remove_world_pool()
        env.databank={}
        env.register={}        
        for i in range(env.sp.V):
            for j in range(0, i):
                if i == env.sp.coord2labels[env.sp.start_escape_route] or j == env.sp.coord2labels[env.sp.start_escape_route]:
                    continue
                if i==j:
                    assert False
                env.redefine_goal_nodes([i,j])
                env.current_entry=i
                env_all.append(copy.deepcopy(env))
    elif scenario_name == 'toptargets-fixed_3U-random-static':
        env.reset()
        # We clip the unit paths to the first position (no movement)
        for i in range(len(env.databank['coords'])):
            patharr=env.databank['coords'][i]['paths']
            for j in range(len(patharr)):
                patharr[j] = [patharr[j][0]]
            patharr=env.databank['labels'][i]['paths']
            for j in range(len(patharr)):
                patharr[j] = [patharr[j][0]]
        env_all = [env]
    elif scenario_name == 'toptargets-fixed_3U-random-dynamic':
        env.reset()
        env_all = [env]
    else:
        assert False
    #SimulateInteractiveMode(env)

    config={}
    config['node_dim']      = env_all[0].F
    config['max_num_nodes']     = env_all[0].sp.V
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
    config['logdir']        = rootdir + \
                                nfm_func.name+'/'+ \
                                '_emb'+str(config['emb_dim']) + \
                                '_itT'+str(config['emb_iter_T']) + \
                                '_epi'+str(config['num_episodes']) + \
                                '_mem'+str(config['memory_size']) + \
                                '_nstep'+str(config['num_step_ql'])
    numseeds=1
    seed0=0
    train(seeds=numseeds,seednr0=seed0, config=config, env_all=env_all)
    evaluate(logdir=config['logdir']+'/SEED'+str(seed0), config=config, env_all=env_all)
    #evaluate_spath_heuristic(logdir=rootdir+'/heur/spath', config, env_all=env_all)
    #evaluate_tabular(logdir=rootdir+'/tabular', config, env_all=env_all)