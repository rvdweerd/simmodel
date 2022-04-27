import copy
import modules.gnn.nfm_gen
from modules.ppo.models_sb3_gat2 import DeployablePPOPolicy_gat2
from modules.ppo.models_sb3_s2v import s2v_ActorCriticPolicy, Struc2VecExtractor, DeployablePPOPolicy
from modules.ppo.ppo_wrappers import PPO_ActWrapper, PPO_ObsWrapper, PPO_ObsDictWrapper, VarTargetWrapper, PPO_ObsFlatWrapper, PPO_ObsBasicDictWrapper, PPO_ObsBasicDictWrapperCRE
from modules.rl.environments import GraphWorld
from modules.rl.rl_utils import EvaluatePolicy, EvalArgs1, EvalArgs2, EvalArgs3, GetFullCoverageSample
from modules.rl.rl_policy import ActionMaskedPolicySB3_PPO
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.environments import SuperEnv
from modules.rl.rl_plotting import PlotEUPathsOnGraph_
from modules.sim.graph_factory import GetWorldSet, LoadData
import numpy as np
import torch.nn as nn 
import torch
import torch.nn.functional as F
import tqdm
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib import MaskablePPO
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_super_env(Uselected=[1], Eselected=[4], config=None, var_targets=None, apply_wrappers=True, type_obs_wrap='Flat',remove_paths=False):
    max_nodes=config['max_nodes']
    max_edges=config['max_edges']
    state_repr = 'etUte0U0'
    state_enc  = 'nfm'
    nfm_func = modules.gnn.nfm_gen.nfm_funcs[config['nfm_func']] 
    edge_blocking = config['edge_blocking']
    solve_select = config['solve_select'] # only solvable worlds (so best achievable performance is 100%)
    reject_u_duplicates = False

    env_all_train, hashint2env, env2hashint, env2hashstr = GetWorldSet(state_repr, state_enc, U=Uselected, E=Eselected, edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_u_duplicates, nfm_func=nfm_func, var_targets=var_targets, remove_paths=remove_paths)
    if apply_wrappers:
        for i in range(len(env_all_train)):
            if type_obs_wrap == 'Flat':
                env_all_train[i]=PPO_ObsFlatWrapper(env_all_train[i], max_possible_num_nodes = max_nodes, max_possible_num_edges=max_edges, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
            elif type_obs_wrap == 'Dict':
                env_all_train[i]=PPO_ObsDictWrapper(env_all_train[i], max_possible_num_nodes = max_nodes, max_possible_num_edges=max_edges)
            else: assert False
            env_all_train[i]=PPO_ActWrapper(env_all_train[i])        
    super_env = SuperEnv(env_all_train, hashint2env, max_possible_num_nodes = max_nodes, max_possible_num_edges=max_edges)
    return super_env, env_all_train

def  CreateEnvFS(config, obs_mask, obs_rate, max_nodes, max_edges):
    Etest=[0,1,2,3,4,5,6,7,8,9,10]
    Utest=[1,2,3]
    evalenv, _, _, _  = GetWorldSet('etUte0U0', 'nfm', U=Utest, E=Etest, edge_blocking=config['edge_blocking'], solve_select=config['solve_select'], reject_duplicates=False, nfm_func=modules.gnn.nfm_gen.nfm_funcs[config['nfm_func']], apply_wrappers=False, maxnodes=world_dict[world_name][0], maxedges=world_dict[world_name][1])
    for i in range(len(evalenv)):
        evalenv[i]=PPO_ObsBasicDictWrapper(evalenv[i], obs_mask=obs_mask, obs_rate=obs_rate)
        evalenv[i]=PPO_ActWrapper(evalenv[i])
    return evalenv

def CreateEnv(world_name, max_nodes=9, max_edges=300, nfm_func_name = 'NFM_ev_ec_t_um_us', var_targets=None, remove_world_pool=False, apply_wrappers=True, type_obs_wrap='Flat', obs_mask='None', obs_rate=1):
    state_repr='etUte0U0'
    state_enc='nfm'
    edge_blocking = True
    env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    env.redefine_nfm(modules.gnn.nfm_gen.nfm_funcs[nfm_func_name])
    env.capture_on_edges = edge_blocking
    if remove_world_pool:
        env._remove_world_pool()
    if var_targets is not None:
        env = VarTargetWrapper(env, var_targets)
    if apply_wrappers:
        #env = PPO_ObsWrapper(env, max_possible_num_nodes = max_nodes)        
        #env = PPO_ObsDictWrapper(env, max_possible_num_nodes = max_nodes, max_possible_num_edges = max_edges)
        if type_obs_wrap == 'Flat':
            env = PPO_ObsFlatWrapper(env, max_possible_num_nodes = max_nodes, max_possible_num_edges=max_edges, obs_mask=obs_mask, obs_rate=obs_rate)
        elif type_obs_wrap == 'Dict':
            env = PPO_ObsDictWrapper(env, max_possible_num_nodes = max_nodes, max_possible_num_edges=max_edges)
        elif type_obs_wrap == 'BasicDict':
            env = PPO_ObsBasicDictWrapper(env, obs_mask=obs_mask, obs_rate=obs_rate)
        elif type_obs_wrap == 'BasicDictCRE':
            env = PPO_ObsBasicDictWrapperCRE(env, obs_mask=obs_mask, obs_rate=obs_rate)            
        else: assert False
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
    print('state:',state,end='')
    W = nn.functional.pad(env.sp.W,(0,p,0,p))
    obs = torch.cat((nfm, W, torch.index_select(W, 1, torch.tensor(epath[-1]))),1)#.to(device)
    reachable=np.zeros(env.sp.V).astype(np.bool)
    idx=np.array(env.neighbors[epath[-1]]).astype(int)
    reachable[idx]=True
    action_masks=list(reachable)+[False]*p
    action_masks=torch.tensor(action_masks,dtype=torch.bool)#.to(device)

    action, _state = ppo_policy.predict(obs, deterministic=True, action_masks=action_masks)
    probs1 = F.softmax(ppo_policy.get_distribution(obs[None].to(device)).log_prob(torch.tensor(env.neighbors[epath[-1]]).to(device)),dim=0)
    ppo_value = ppo_policy.predict_values(obs[None].to(device)).detach().cpu().numpy()
    np.set_printoptions(formatter={'float':"{0:0.2f}".format})
    print('; estimated  value of graph state:',ppo_value)
    print('actions',[a for a in env.neighbors[epath[-1]]],'; probs:',probs1.detach().cpu().numpy(), 'chosen:', action)
    newsp=copy.deepcopy(env.sp)
    newsp.target_nodes=targetnodes
    
    epath0 = [epath[0]]
    upath0 = [[u[0]] for u in upaths]
    fname=logdir+'/'+'hashint'+str(hashint)+'_target'+str(targetnodes)+'_epath'+str(epath)+'_upaths'+str(upaths)
    PlotEUPathsOnGraph_(newsp,epath0,upath0,filename=fname,fig_show=False,fig_save=True,goal_reached=False,last_step_only=False)
    #epath+=[action]
    fname=logdir+'/'+'hashint'+str(hashint)+'_target'+str(targetnodes)+'_epath'+str(epath)+'_upaths'+str(upaths)
    PlotEUPathsOnGraph_(newsp,epath,upaths,filename=fname,fig_show=False,fig_save=True,goal_reached=(epath[-1] in targetnodes),last_step_only=False)

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

def evaluate_ppo(logdir, info=False, config=None, env=None, eval_subdir='.', n_eval=1000, max_num_nodes=-1):
    if env==None: return 0,0,0,0   
    try:
        if env==None or len(env)==0:
            return 0,0,0,0
    except:
        pass
    evaldir=logdir+'/'+eval_subdir
    R=[]
    S=[]
    
    if type(env) == list:
        if len(env)==0: return 0,0,0,0
        saved_model = MaskablePPO.load(logdir+"/saved_models/model_best")
        if config['qnet']=='s2v':
            saved_policy_deployable=DeployablePPOPolicy(env[0], saved_model.policy)
        elif config['qnet']=='gat2':
            saved_policy_deployable=DeployablePPOPolicy_gat2(env[0], saved_model.policy, max_num_nodes=max_num_nodes)
        ppo_policy = ActionMaskedPolicySB3_PPO(saved_policy_deployable, deterministic=True)
        for i,e in enumerate(tqdm.tqdm(env)):

            l, returns, c, solves = EvaluatePolicy(e, ppo_policy, e.world_pool, print_runs=False, save_plots=False, logdir=evaldir, eval_arg_func=EvalArgs3, silent_mode=True)
            num_worlds_requested = 10
            once_every = max(1,len(env)//num_worlds_requested)
            if i % once_every ==0:
                plotlist = GetFullCoverageSample(returns, e.world_pool, bins=3, n=3)
                EvaluatePolicy(e, ppo_policy, plotlist, print_runs=True, save_plots=True, logdir=evaldir, eval_arg_func=EvalArgs3, silent_mode=False, plot_each_timestep=False)
            R+=returns 
            S+=solves

    else:
        # assume env_all is a superenv
        full_eval_list = [i for i in range(n_eval)]
        plot_eval_list = [i for i in range(30)]
        saved_policy = s2v_ActorCriticPolicy.load(logdir+"/saved_models/policy_last")
        saved_policy_deployable=DeployablePPOPolicy(env, saved_policy)
        ppo_policy = ActionMaskedPolicySB3_PPO(saved_policy_deployable, deterministic=True)

        l, returns, c, solves = EvaluatePolicy(env, ppo_policy, full_eval_list, print_runs=False, save_plots=False, logdir=logdir, eval_arg_func=EvalArgs3, silent_mode=True)
        EvaluatePolicy(env, ppo_policy, plot_eval_list, print_runs=True, save_plots=True, logdir=logdir, eval_arg_func=EvalArgs3, silent_mode=False, plot_each_timestep=False)
        R+=returns 
        S+=solves
    
    OF = open(logdir+'/Full_result.txt', 'w')
    def printing(text):
        print(text)
        OF.write(text + "\n")
    num_unique_graphs=-1#len(env_all)
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

def evaluate_lstm_ppo(logdir, info=False, config=None, env=None, ppo_policy=None, eval_subdir='.', n_eval=1000, max_num_nodes=-1, multiplier=1):
    if env==None: return 0,0,0,0   
    try:
        if env==None or len(env)==0:
            return 0,0,0,0
    except:
        pass
    evaldir=logdir+'/'+eval_subdir
    R=[]
    S=[]
    
    if type(env) == list:
        if len(env)==0: return 0,0,0,0
        for i,e in enumerate(tqdm.tqdm(env)):
            

            l, returns, c, solves = EvaluatePolicy(e, ppo_policy, e.world_pool * multiplier, print_runs=False, save_plots=False, logdir=evaldir, eval_arg_func=EvalArgs3, silent_mode=True)
            num_worlds_requested = 10
            once_every = max(1,len(env)//num_worlds_requested)
            if i % once_every ==0:
                plotlist = GetFullCoverageSample(returns, e.world_pool * multiplier, bins=3, n=3)
                EvaluatePolicy(e, ppo_policy, plotlist, print_runs=True, save_plots=True, logdir=evaldir, eval_arg_func=EvalArgs3, silent_mode=False, plot_each_timestep=False)
            R+=returns 
            S+=solves
    else:
        # assume env_all is a superenv
        assert False
    
    OF = open(evaldir+'/Full_result.txt', 'w')
    def printing(text):
        print(text)
        OF.write(text + "\n")
    num_unique_graphs=-1#len(env_all)
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