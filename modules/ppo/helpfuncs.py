import numpy as np
import torch.nn as nn 
import torch
import torch.nn.functional as F
import copy
import tqdm
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from modules.rl.rl_utils import EvaluatePolicy, EvalArgs1, EvalArgs2, GetFullCoverageSample
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us, NFM_ev_ec_t_u
from modules.ppo.ppo_wrappers import PPO_ActWrapper, PPO_ObsWrapper, VarTargetWrapper
from modules.sim.graph_factory import GetWorldSet, LoadData
from modules.rl.environments import GraphWorld
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.environments import SuperEnv
from modules.rl.rl_plotting import PlotEUPathsOnGraph_
device = 'cuda' if torch.cuda.is_available() else 'cpu'


nfm_funcs = {
    'NFM_ev_ec_t'       : NFM_ev_ec_t(),
    'NFM_ec_t'          : NFM_ec_t(),
    'NFM_ev_t'          : NFM_ev_t(),
    'NFM_ev_ec_t_u'     : NFM_ev_ec_t_u(),
    'NFM_ev_ec_t_um_us' : NFM_ev_ec_t_um_us(),
    }

def get_super_env(Uselected=[1], Eselected=[4], config=None, var_targets=None):
    # scenario_name=config['scenario_name']
    #scenario_name = 'Train_U2E45'
    # world_name = 'SubGraphsManhattan3x3'
    max_nodes=config['max_nodes']
    state_repr = 'etUte0U0'
    state_enc  = 'nfm'
    nfm_funcs = {'NFM_ev_ec_t':NFM_ev_ec_t(),'NFM_ec_t':NFM_ec_t(),'NFM_ev_t':NFM_ev_t(),'NFM_ev_ec_t_um_us':NFM_ev_ec_t_um_us()}
    nfm_func=nfm_funcs[config['nfm_func_name']]
    edge_blocking = config['edge_blocking']
    solve_select = config['solve_select']# only solvable worlds (so best achievable performance is 100%)
    reject_u_duplicates = False

    #databank_full, register_full, solvable = LoadData(edge_blocking = True)
    env_all_train, hashint2env, env2hashint, env2hashstr = GetWorldSet(state_repr, state_enc, U=Uselected, E=Eselected, edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_u_duplicates, nfm_func=nfm_func, var_targets=var_targets)
    for i in range(len(env_all_train)):
        env_all_train[i]=PPO_ObsWrapper(env_all_train[i], max_possible_num_nodes = max_nodes)        
        env_all_train[i]=PPO_ActWrapper(env_all_train[i])        
    super_env = SuperEnv(env_all_train, hashint2env, max_possible_num_nodes = max_nodes)
    #SimulateInteractiveMode(super_env)
    return super_env, env_all_train

def CreateEnv(world_name, max_nodes=9, var_targets=None, remove_world_pool=False):
    #world_name='Manhattan3x3_WalkAround'
    state_repr='etUte0U0'
    state_enc='nfm'
    nfm_func_name = 'NFM_ev_ec_t_um_us'
    edge_blocking = True
    #remove_world_pool = False
    env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    env.redefine_nfm(nfm_funcs[nfm_func_name])
    env.capture_on_edges = edge_blocking
    if remove_world_pool:
        env._remove_world_pool()
    #SimulateInteractiveMode(env,filesave_with_time_suffix=False)
    #MAX_NODES=9
    if var_targets is not None:
        env = VarTargetWrapper(env, var_targets)
    env = PPO_ObsWrapper(env, max_possible_num_nodes = max_nodes)        
    env = PPO_ActWrapper(env)        
    return env

def check_custom_position_probs(env,ppo_policy,hashint=None,entry=None,targetnodes=[1],epath=[4],upaths=[[6]],max_nodes=0,logdir='test'):    
    if hashint is not None:
        obs=env.reset(hashint, entry)
    else:
        obs=env.reset(entry=entry)
    nfm, state = env.get_custom_nfm(epath,targetnodes,upaths)
    p = max_nodes - env.sp.V
    nfm = nn.functional.pad(nfm,(0,0,0,p))
    print('state:',state)
    #print(nfm)
    W = nn.functional.pad(env.sp.W,(0,p,0,p))
    obs = torch.cat((nfm, W, torch.index_select(W, 1, torch.tensor(epath[-1]))),1)#.to(device)
    reachable=np.zeros(env.sp.V).astype(np.bool)
    idx=np.array(env.neighbors[epath[-1]]).astype(int)
    reachable[idx]=True
    action_masks=list(reachable)+[False]*p
    action_masks=torch.tensor(action_masks,dtype=torch.bool)#.to(device)

    action, _state = ppo_policy.predict(obs, deterministic=True, action_masks=action_masks)
    probs1 = F.softmax(ppo_policy.get_distribution(obs[None].to(device)).log_prob(torch.tensor(env.neighbors[epath[-1]]).to(device)),dim=0)
    np.set_printoptions(formatter={'float':"{0:0.2f}".format})
    print('actions',[a for a in env.neighbors[epath[-1]]],'; probs:',probs1.detach().cpu().numpy(), 'chosen:', action)

    newsp=copy.deepcopy(env.sp)
    newsp.target_nodes=targetnodes
    fname=logdir+'/'+'hashint'+str(hashint)+'_target'+str(targetnodes)+'_epath'+str(epath)+'_upaths'+str(upaths)
    PlotEUPathsOnGraph_(newsp,epath,upaths,filename=fname,fig_show=False,fig_save=True,goal_reached=False,last_step_only=False)

def eval_simple(saved_policy,env):
    res_det = evaluate_policy(saved_policy, env, n_eval_episodes=20, reward_threshold=-100, warn=False, return_episode_rewards=False, deterministic=True)
    print('Test result (deterministic): avg rew:', res_det[0], 'std:', res_det[1])
    res_nondet = evaluate_policy(saved_policy, env, n_eval_episodes=20, reward_threshold=-100, warn=False, return_episode_rewards=False, deterministic=False)
    print('Test result (non-deterministic): avg rew:', res_nondet[0], 'std:', res_nondet[1])

    np.set_printoptions(formatter={'float':"{0:0.2f}".format})
    fpath='results/gnn-ppo/sb3/'
    for i in range(10):
        obs = env.reset()
        print('\nHashint: ',env.sp.hashint, ', entry:', env.current_entry)
        print('Initial state:',env.state)
        env.env.render(fname=fpath+'test'+str(i))
        done=False
        while not done:
            action_masks = get_action_masks(env)
            action, _state = saved_policy.predict(obs, deterministic=True, action_masks=action_masks)
            probs1 = F.softmax(saved_policy.get_distribution(obs[None].to(device)).log_prob(torch.tensor(env.neighbors[env.state[0]]).to(device)),dim=0)
            print('actions',[a for a in env.neighbors[env.state[0]]],'; probs:',probs1.detach().cpu().numpy(), 'chosen:', action)
            obs, reward, done, info = env.step(action)
            env.env.render(fname=fpath+'test'+str(i))
            if done:
                env.env.render_eupaths(fname=fpath+'test'+str(i)+'_final')
                obs = env.reset()


def evaluate_ppo(logdir, policy, info=False, config=None, env_all=None, eval_subdir='.'):
    #Test(config)
    if env_all==None or len(env_all)==0:
        return 0,0,0,0
    
    logdir=logdir+'/'+eval_subdir
    #Q_func, Q_net, optimizer, lr_scheduler = init_model(config,fname='results_Phase2/CombOpt/'+affix+'/ep_350_length_7.0.tar')
    #policy.epsilon=0.

    R=[]
    S=[]
    for i,env in enumerate(tqdm.tqdm(env_all)):
        l, returns, c, solves = EvaluatePolicy(env, policy,env.world_pool, print_runs=False, save_plots=False, logdir=logdir, eval_arg_func=EvalArgs2, silent_mode=True)
        num_worlds_requested = 10
        once_every = max(1,len(env_all)//num_worlds_requested)
        if i % once_every ==0:
            plotlist = GetFullCoverageSample(returns, env.world_pool, bins=3, n=3)
            EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir=logdir, eval_arg_func=EvalArgs2, silent_mode=False, plot_each_timestep=False)
        R+=returns 
        S+=solves
    
    OF = open(logdir+'/Full_result.txt', 'w')
    def printing(text):
        print(text)
        OF.write(text + "\n")
    num_unique_graphs=len(env_all)
    num_graph_instances=len(R)
    avg_return=np.mean(R)
    num_solved=np.sum(S)
    success_rate = num_solved/len(S)
    printing('Total unique graphs evaluated: '+str(num_unique_graphs))
    printing('Total instances evaluated: '+str(num_graph_instances)+' Avg reward: {:.2f}'.format(avg_return))
    printing('Goal reached: '+str(num_solved)+' ({:.1f}'.format(success_rate*100)+'%)')
    printing('---------------------------------------')
    for k,v in config.items():
        printing(k+' '+str(v))
    return num_unique_graphs, num_graph_instances, avg_return, success_rate

def get_logdirs(config):
    rootdir = 'results/results_Phase2/Pathfinding/ppo/'+ \
                config['train_on'] + \
                '/solvselect=' + config['solve_select']+'_edgeblock='+str(config['edge_blocking'])+'/' +\
                config['scenario_name']
    logdir = rootdir+'/'+ \
                config['nfm_func_name'] +'/'+ \
                's2v_layers='+str(config['s2v_layers']) + \
                '_emb='+str(config['emb_dim']) + \
                '_itT='+str(config['emb_iter_T']) + \
                '_nstep='+str(config['num_step'])
    return rootdir, logdir