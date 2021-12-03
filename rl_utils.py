from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda import init

def EvaluatePolicy(env, policy, test_set, print_runs=True, save_plots=False):
    # Escaper chooses random neighboring nodes until temination
    # Inputs:
    #   test_set: list of indices to the databank
    captured=[]
    iratios_sampled=[]
    rewards=[]
    lengths=[]
    for i, entry in enumerate(test_set):
        s=env.reset(entry)
        iratios_sampled.append(env.iratio)
        done=False
        R=0
        if env.sp.coord2labels[env.sp.target_node] == env.state[0]:
            done = True
            R=10
            info = {'Captured':True}
        if print_runs:
            print('Run',i+1,': Initial state:,',env.state0,', Path:[',end='')
        count=0
        #e_history=[]
        while not done:
            if save_plots:
                env.render(file_name='images_rl/Run'+str(i+1)+'_s0='+str(env.state0)+'t='+str(env.global_t))
            #e_history.append(s[0])
            if print_runs:
                print(str(env.state[0])+'->',end='')
            
            action,_ = policy.sample_action(s, env._availableActionsInCurrentState())

            s,r,done,info = env.step(action)
            count+=1
            R+=r
            # if count >= env.sp.L:
            #     break
        if print_runs:
            print(str(s[0])+']. Done after',count,'steps, Captured:',info['Captured'],'Reward:',str(R))
        if save_plots:
            env.render(file_name='images_rl/Run'+str(i+1)+'_s0='+str(env.state0)+'t='+str(env.global_t))
        captured.append(int(info['Captured']))
        rewards.append(R)
        lengths.append(count)
        plt.clf()
    print('------------------')
    print('Test set size:',len(test_set),'Observed escape ratio: {:.3f}'.format(1-np.mean(captured)),', Average episode length: {:.2f}'.format(np.mean(lengths)),', Average return: {:.2f}'.format(np.mean(rewards)))
    print('Escape ratio at data generation: last {:.3f}'.format(1-env.iratio),', avg at generation {:.3f}'.format(1-sum(env.iratios)/len(env.iratios)),\
        ', avg sampled {:.3f}'.format(1-sum(iratios_sampled)/len(iratios_sampled)),'\n')
    if len(rewards) <20:
        print('Returns:',rewards)

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