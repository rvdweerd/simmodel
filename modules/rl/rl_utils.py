from itertools import combinations
import torch
from stable_baselines3 import PPO
import gym
import numpy as np
import matplotlib.pyplot as plt
#from torch.cuda import init
from pathlib import Path
from torch_geometric.data import Data, Batch

plt.switch_backend('agg')

def CalculateNumTrainableParameters(model):
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
    assert total == sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total

def EvalArgs1(env):
    # arguments of the call to the policy class that samples an action
    # this is the default
    return env.obs, env.availableActionsInCurrentState()

def EvalArgs2(env):
    # arguments of the call to the policy class that samples an action
    # this version is used for GNNs
    return env.nfm, env.sp.W, Data(env.nfm, env.sp.EI), env.neighbors[env.state[0]]

def EvalArgs3(env):
    # arguments of the call to the policy class that samples an action
    # this is used when observation wrappers are used
    return env.observation(None), env.availableActionsInCurrentState()

def EvalArgsSP(env):
    # return next node in shortest path and current node
    if len(env.sp.spath_to_target) == 1:
        return (env.neighbors[env.state[0]][0], env.state[0]), None
    return (env.sp.spath_to_target[env.global_t+1], env.state[0]), None


def EvaluatePolicy(env, policy, test_set, print_runs=True, save_plots=False, logdir='./temp', has_Q_table=False, eval_arg_func=EvalArgs1, silent_mode=False, plot_each_timestep=True):
    # Escaper chooses random neighboring nodes until temination
    # Inputs:
    #   test_set: list of indices to the databank
    captures=[]
    iratios_sampled=[]
    lengths=[]
    returns=[]
    solves=[]
    #if print_runs or save_plots:
    Path(logdir).mkdir(parents=True, exist_ok=True)
    Path(logdir+'/runs').mkdir(parents=True, exist_ok=True)
    if env.sp.hashint != -1:
        hint=str(env.sp.hashint)+'_'
    else:
        hint=''
    file_prefix=logdir+'/runs/'+hint+'Entry='
    OutputFile= logdir+'/runs/'+hint+'Log_n='+str(len(test_set))+'.txt'
    if silent_mode:
        def printing(text):
            pass
    else:
        OF = open(OutputFile, 'w')
        def printing(text):
            print(text)
            OF.write(text + "\n")
    np.set_printoptions(formatter={'float':"{0:0.1f}".format})
    np.set_printoptions(formatter={'int'  :"{0:<3}".format})
    if has_Q_table:
        count=0
        for k,v in policy.Q.items():
            for i in v:
                count+=1
        printing('entries in Q table: '+str(len(policy.Q)))
        printing('Total number of q values stored: '+str(count))
    else:
        if issubclass(type(policy.model),torch.nn.Module) and not silent_mode:
            printing(str(policy.model))
            printing('#Trainable parameters: '+str(policy.model.numTrainableParameters()))
        if issubclass(type(policy.model),PPO) and not silent_mode:
            printing(str(policy.model.policy))
            printing('#Trainable parameters: '+str(CalculateNumTrainableParameters(policy.model.policy)))
        printing('\n-------------------------------------------------------------------------------------------------------')   
    if len(test_set)==0: test_set=[None]
    MAX_TEST_SET_SIZE = 65000
    if len(test_set) > MAX_TEST_SET_SIZE:
        test_set=list(np.random.choice(test_set,size=MAX_TEST_SET_SIZE,replace=False))
    for i, entry in enumerate(test_set):
        s=env.reset(entry = entry) 
        policy.reset_hidden_states(env)
        iratios_sampled.append(env.iratio)
        done=False
        text_cache=""
        plot_cache=[]
        R=0
        count=0
        if len(env.sp.target_nodes)>0 and env.state[0] in env.sp.target_nodes:
            done = True
            R=10
            info = {'Captured':False, 'Solved':True}
        if print_runs:
            #printing('\nRun '+str(i+1)+': dataset entry '+str(entry)+', Initial state '+str(env.state0))
            text_cache += ('\nRun '+str(i+1)+': dataset entry '+str(entry)+', Initial state '+str(env.state0))
            text_cache += ', Deterministic policy: '+str(policy.deterministic)+'\n'
        while not done:
            if save_plots and plot_each_timestep:
                plot=env.render_eupaths(fname=None, last_step_only=True)
                plot_cache.append(plot)
            s_start=env.state
            #action,_ = policy.sample_action(s, env.availableActionsInCurrentState())
            #action,_ = policy.sample_action(env.nfm, env.sp.W, env.neighbors[env.state[0]])
            with torch.no_grad():
                action,_ = policy.sample_action(*eval_arg_func(env))
            action_probs = policy.get_action_probs()
            s,r,done,info = env.step(action)
            s_prime = env.state
            if print_runs:
                np.set_printoptions(formatter={'float':"{0:0.1f}".format})
                text_cache+=('  s: {0:>18}'.format(str(s_start))+' a '+str(action)+' action_probs '+str(action_probs)+' r '+str(r)+' s_ '+str(s_prime)+'\n')
            count+=1
            R+=r
        if print_runs:
            text_cache+=('  Done after '+str(count)+' steps, Captured: '+str(info['Captured'])+' Reward: '+str(R)+'\n')
            printing(text_cache)
        if save_plots:
            if plot_each_timestep:
                plot=env.render_eupaths(fname=None, last_step_only=True)
                plot_cache.append(plot)
            for i_plt,p in enumerate(plot_cache):
                fname=file_prefix+str(env.current_entry)+'_s0='+str(env.state0)+'_'+policy.__name__+'_R='+str(R)+'_t='+str(i_plt)
                p.savefig(fname+'.png')
            
            fname=file_prefix+str(env.current_entry)+'_s0='+str(env.state0)+'_'+policy.__name__+'_R='+str(R)+'_fulleupaths'
            env.render_eupaths(fname=fname)
            #plot2 = env.render_epath(fname=None)
            #plot2.savefig(fname+'.png')
        captures.append(int(info['Captured']))
        solves.append(int(info['Solved']))
        returns.append(R)
        lengths.append(count)
    printing('\nAggregated test results.')
    printing('  > Environment : '+env.sp.graph_type+'_N='+str(env.sp.N)+'_U='+str(env.sp.U)+'_T='+str(env.max_timesteps)+'_Ndir='+str(env.sp.direction_north)[0])
    printing('  > Policy      : '+policy.__name__+', Deterministic: '+str(policy.deterministic))
    printing('Test set size: '+str(len(test_set)))
    printing('Observed escape ratio: {:.3f}'.format(1-np.mean(captures)))
    if len(returns) <20 or print_runs:
        printing('   Captures:'+str(np.array(captures)))
    num_solves=np.sum(solves)
    solve_ratio=(num_solves/len(solves))
    printing('Goal reached: {:.0f}'.format(num_solves)+'/{:.0f}'.format(len(solves))+', solve ratio: {:.3f}'.format(solve_ratio))
    printing('Average episode length: {:.2f}'.format(np.mean(lengths))+' +/- {:.2f}'.format(np.std(lengths)))
    if len(returns) <20 or print_runs:
        printing('   Lengths :'+str(np.array(lengths) ))
    printing('Average return: {:.2f}'.format(np.mean(returns))+' +/- {:.2f}'.format(np.std(returns)))
    if len(returns) <20 or print_runs:
        printing('   Returns :'+str(np.array(returns) ))
    printing('\nEscape ratio at data generation: last {:.3f}'.format(1-env.iratio)+', avg at generation {:.3f}'.format(1-sum(env.iratios)/len(env.iratios))+\
        ', avg sampled {:.3f}'.format(1-sum(iratios_sampled)/len(iratios_sampled)))
    printing('-------------------------------------------------------------------------------------------------------')
    return (lengths, returns, captures, solves)

def GetInitialStatesList(env, min_y_coord):
    # Get pointers to all initial conditions with Units above min_y_coord
    idx=[]
    initpos=[]
    for k,v in env.register['coords'].items():
        if k[0][1]==0:
            valid = True
            for u in k[1:]:
                if u[1] < min_y_coord:
                    valid = False
                    break
            if valid:
                idx.append(v)
                ipos=[env.sp.coord2labels[k[0]]]
                for u in k[1:]:
                    ipos.append(env.sp.coord2labels[u])
                ipos.sort()
                initpos.append(tuple(ipos))
    return idx, initpos

def GetSetOfPermutations(path):
    out=set()
    for n in range(len(path) + 1):
        for i in combinations(path,n):
            if len(i)==0: continue
            j=list(i)
            j.sort()
            out.add(tuple(j))
    return out

def GetDuplicatePathsIndices(paths, min_num_same_positions, min_num_worlds, print_selection=False):
    duplicates_idx=set()
    for row in range(paths.shape[1]):
        map={}
        map2envs={}
        map2time={}
        for n in range(paths.shape[0]):
            perms=GetSetOfPermutations(paths[n][row])
            for p in perms:
                if p not in map2envs:
                    map2envs[p]=[n]
                else:
                    map2envs[p].append(n)
                if p not in map2time:
                    map2time[p]=[row]
                else:
                    map2time[p].append(row)                
                if p not in map:
                    map[p]=1
                else:
                    map[p]+=1
        for k,v in map.items():
            if len(k) >= min_num_same_positions and v >= min_num_worlds: # at least 3 units on same position, at least in 2 worlds
                if print_selection:
                    print('identical U positions',k,'appear in worlds',map2envs[k],'in time-step',row)
                for i in map2envs[k]:
                    duplicates_idx.add(i)
    return list(duplicates_idx)

def GetPathsTensor(env, db_indices, print_selection=False):
    paths = []
    for i in db_indices:
        e = env.databank['labels'][i]
        #print(  'start_escape_route',e['start_escape_route'],\
        #        'start_units',e['start_units'],\
        #        'paths',e['paths'])
        unit_paths=e['paths']
        ipaths = []
        for t in range(env.sp.L):
            p = []
            for P_path in unit_paths:
                pos = P_path[-1] if t >= len(P_path) else P_path[t]
                p.append(pos)
            ipaths.append(tuple(p))
        paths.append(ipaths)
    paths_np=np.array(paths)
    if print_selection:
        print('Paths found:',paths_np.shape)
    return paths_np, paths

def CreateDuplicatesTrainsets(env, min_y_coord, min_num_same_positions, min_num_worlds, print_selection=False):
    db_indices, init_pos_list = GetInitialStatesList(env, min_y_coord)
    paths_np, paths = GetPathsTensor(env, db_indices, print_selection=print_selection)

    duplicates_indices0=set()
    for world_source in range(len(paths)):
        for i in range(len(paths[world_source])):
            key1=list(paths[world_source][i])
            key1.sort()
            for world_target in range(len(paths)):
                if world_target==world_source: continue
                for t in range(env.sp.L-1):
                    key2_t=list(paths[world_target][t])
                    key2_t.sort()
                    key2_tt=list(paths[world_target][t+1])
                    key2_tt.sort()
                    #if key2_t[:2] == key1[:2] and key2_tt[:2] != key1[:2]:
                    if key2_t == key1 and key2_tt != key1:
                        if print_selection:
                            print('t=',t,'Pair found:',paths[world_source],paths[world_target])
                        duplicates_indices0.add(world_target)
                        duplicates_indices0.add(world_source)
    
    duplicates_indices0 = list(duplicates_indices0)
    duplicates_indices1 = GetDuplicatePathsIndices(paths_np, min_num_same_positions, min_num_worlds, print_selection=print_selection)    
    
    init_pos_trainset0=[init_pos_list[i] for i in duplicates_indices0]
    init_pos_trainset_indices0=[db_indices[i] for i in duplicates_indices0]
    init_pos_trainset1=[init_pos_list[i] for i in duplicates_indices1]
    init_pos_trainset_indices1=[db_indices[i] for i in duplicates_indices1]

    # remove entries in trainset1 from trainset0 to enhance differences
    for e1 in init_pos_trainset_indices1:
        if e1 in init_pos_trainset_indices0:
            init_pos_trainset_indices0.remove(e1)
    return init_pos_trainset_indices0, init_pos_trainset_indices1

def GetOutliersSample(returns, world_list):
    MAX_INIT = -1e6
    MIN_INIT = 1e6
    maxR=MAX_INIT
    minR=MIN_INIT
    selection=[]
    for i,R in enumerate(returns):
        if R > maxR:
            maxR = R
            if minR == MIN_INIT:
                minR = R
            selection.append(world_list[i])
        if R < minR:
            minR = R
            if maxR == MAX_INIT:
                maxR = R
            selection.append(world_list[i])
    selection.sort()
    return selection

def GetFullCoverageSample(returns, world_list, bins=10, n=10):
    if n>len(returns):
        return world_list
    assert n>=bins
    chosen_worlds=[]
    edges=np.histogram_bin_edges(returns,bins=bins)
    bin_digits=np.digitize(returns,edges)
    a=np.min(bin_digits)
    b=np.max(bin_digits)
    while len(chosen_worlds)<n:
        for bin in np.arange(a,b+1,1):
            indices=np.where(bin_digits==bin)[0]
            if len(indices)>0:
                chosen_index = np.random.choice(indices)
                chosen_worlds.append(world_list[chosen_index])
            if len(chosen_worlds)==n:
                break
    return chosen_worlds


def print_parameters(model):
    print(model)
    print('Policy model size:')
    print('------------------------------------------')
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        print("{:44s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
    print("Total number of parameters: {}".format(total))
    print('------------------------------------------')
    assert total == sum(p.numel() for p in model.parameters() if p.requires_grad)

class NpWrapper(gym.ObservationWrapper):
    def availableActionsInCurrentState(self):
        return None
    def observation(self, observation):
        obs = np.array(observation).astype(np.float64)
        return obs
