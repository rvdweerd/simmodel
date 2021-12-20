from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.cuda import init

def EvaluatePolicy(env, policy, test_set, print_runs=True, save_plots=False):
    # Escaper chooses random neighboring nodes until temination
    # Inputs:
    #   test_set: list of indices to the databank
    captured=[]
    iratios_sampled=[]
    rewards=[]
    lengths=[]
    file_prefix='images/rl/'+env.sp.graph_type+'_Entry='
    maxR=1e-6
    minR=1e6
    OutputFile= 'images/rl/'+env.sp.graph_type+'log_n='+str(len(test_set))+'.txt'
    OF = open(OutputFile, 'w')

    def printing(text):
        print(text)
        OF.write(text + "\n")
    
    np.set_printoptions(formatter={'float':"{0:0.1f}".format})
    printing('\n-------------------------------------------------------------------------------------------------------')
    for i, entry in enumerate(test_set):
        s=env.reset(entry=entry)
        policy.reset_hidden_states()
        iratios_sampled.append(env.iratio)
        done=False
        R=0
        if len(env.sp.target_nodes)>0 and env.state[0] in env.sp.target_nodes:
            done = True
            R=10
            info = {'Captured':False}
        text_cache=""
        plot_cache=[]
        if print_runs:
            #printing('\nRun '+str(i+1)+': dataset entry '+str(entry)+', Initial state '+str(env.state0))
            text_cache += ('\nRun '+str(i+1)+': dataset entry '+str(entry)+', Initial state '+str(env.state0)+'\n')
        count=0
        while not done:
            if save_plots:
                plot=env.render(fname=None)
                plot_cache.append(plot)
            s_start=env.state
            action,_ = policy.sample_action(s, env._availableActionsInCurrentState())
            action_probs = policy.get_action_probs()
            s,r,done,info = env.step(action)
            s_prime = env.state
            if print_runs:
                text_cache+=('  s: {0:>18}'.format(str(s_start))+' a '+str(action)+' action_probs '+str(action_probs)+' r '+str(r)+' s_ '+str(s_prime)+'\n')
            count+=1
            R+=r
        if print_runs:
            text_cache+=('  Done after '+str(count)+' steps, Captured: '+str(info['Captured'])+' Reward: '+str(R)+'\n')
        if save_plots:
            plot=env.render(fname=None)
            plot_cache.append(plot)
        if R>maxR:
            maxR=R
            if minR==1e6: minR=R
            if print_runs:
                printing(text_cache)
            if save_plots:
                for i_plt,p in enumerate(plot_cache):
                    fname=file_prefix+str(entry)+'_s0='+str(env.state0)+'_'+policy.__name__+'_t='+str(i_plt)
                    p.savefig(fname)
                    #plt.clf()
        if R<minR:
            minR=R
            if maxR==-1e6: maxR=R
            if print_runs:
                printing(text_cache)
            if save_plots:
                for i_plt,p in enumerate(plot_cache):
                    fname=file_prefix+str(entry)+'_s0='+str(env.state0)+'_'+policy.__name__+'_t='+str(i_plt)
                    p.savefig(fname)
                    #plt.clf()
        captured.append(int(info['Captured']))
        rewards.append(R)
        lengths.append(count)
    printing('\nAggregated test results:')
    printing('  > Environment : '+env.sp.graph_type+'_N='+str(env.sp.N)+'_U='+str(env.sp.U)+'_T='+str(env.max_timesteps)+'_Ndir='+str(env.sp.direction_north)[0])
    printing('  > Policy      : '+policy.__name__)
    printing('Test set size: '+str(len(test_set))+' Observed escape ratio: {:.3f}'.format(1-np.mean(captured))+', Average episode length: {:.2f}'.format(np.mean(lengths))+', Average return: {:.2f}'.format(np.mean(rewards)))
    printing('Escape ratio at data generation: last {:.3f}'.format(1-env.iratio)+', avg at generation {:.3f}'.format(1-sum(env.iratios)/len(env.iratios))+\
        ', avg sampled {:.3f}'.format(1-sum(iratios_sampled)/len(iratios_sampled)))
    if len(rewards) <20:
        printing('Returns:'+str(rewards))
    #print('-------------------------------------------------------------------------------------------------------')

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