import simdata_utils as su
from environments import GraphWorld
import random


def SimulatePursuersPathways(conf, dataframe=0):
    # Escaper position static, plot progression of pursuers motion
    sp = su.DefineSimParameters(conf)
    dirname = su.make_result_directory(sp)
    register, databank, iratios = su.LoadDatafile(dirname)
    start_escape_node = databank[dataframe]['start_escape_route']
    unit_paths = databank[dataframe]['paths']
    print('unit_paths',unit_paths,'intercept_rate',iratios[dataframe])
    for t in range(sp.L):
        e = start_escape_node
        p = []
        for P_path in unit_paths:
            pos = P_path[-1] if t >= len(P_path) else P_path[t]
            p.append(pos)
        su.PlotAgentsOnGraph(sp, e, p, t)

def SimulateInteractiveMode(conf):
    env=GraphWorld(conf,optimization='static')
    #s=env.reset(((1,0),(0,2),(2,2)))
    s=env.reset()#((2,0),(2,3),(3,4),(4,3)))
    done=False
    R=0
    env.render()
    while not done:
        print('u paths:',env.u_paths,'\nu positions:')
        for t in range(5):
            print(env._getUpositions(t))
        print('------------')
        print('Current state:')
        print(s)
        print('Available actions:')
        for k,v in env._availableActionsInCurrentState().items():
            print('>',k,v)
        dir=input('Action (new node)?\n> ')
        s,r,done,_=env.step(dir)
        env.render()
        R+=r
    print('done, reward='+str(R),'\n---------------')

def SimulateRandomWalker(conf, number_of_runs=1, optimization_method='static', print_runs=True, save_plots=False):
    # Escaper chooses random neighboring nodes until temination
    # Inputs:
    #   optimization_method: if dynamic, new optimal unit position targets are used at every time-step
    #                        if static, only optimal unit position targets calculated at start are used and fixed
    env=GraphWorld(conf,optimization=optimization_method)
    #s=env.reset(((1,0),(0,2),(2,2)))
    captured=[]
    iratios_sampled=[]
    rewards=[]
    for i in range(number_of_runs):
        s=env.reset()#((2,0),(0,4),(2,4),(4,4)))
        iratios_sampled.append(env.iratio)
        done=False
        R=0
        if print_runs:
            print('Run',i+1,": [",end='')
        count=0
        #e_history=[]
        while not done:
            #e_history.append(s[0])
            if save_plots:
                env.render()
            if print_runs:
                print(str(s[0])+'->',end='')
            #Random policy
            possible_actions = env._availableActionsInCurrentState()['coords']
            action = random.choice(possible_actions)

            # Always all the way left, then all the way up
            # possible_actions = env._availableActionsInCurrentState()['node_labels']
            # n=env.sp.coord2labels[s[0]]
            # if n-1 in possible_actions: # can go left
            #     action = env.sp.labels2coord[n-1]
            #     #print('L',end='')
            # elif n+env.sp.N in possible_actions: # can't go left, go up
            #     action = env.sp.labels2coord[n+env.sp.N]
            #     #print('U',end='')
            # elif n+1 in possible_actions: # can't go up, go right
            #     action = env.sp.labels2coord[n+1]
            #     #print('R',end='')
            # else: # go down
            #     action = env.sp.labels2coord[n-env.sp.N]
            #     #print('D',end='')

            s,r,done,info = env.step(action)
            count+=1
            R+=r
            if count >= env.sp.L:
                break
        if print_runs:
            print(str(s[0])+']. Done after',count,'steps, Captured:',info['Captured'],'Reward:',str(R))
        if save_plots:
            env.render()
        captured.append(int(info['Captured']))
        rewards.append(R)
    print('------------------')
    print('Observed capture ratio: ',sum(captured)/len(captured),', Average reward:',sum(rewards)/len(rewards))
    print('Capture ratio at data generation: last',env.iratio,' avg at generation',sum(env.iratios)/len(env.iratios),\
        'avg sampled',sum(iratios_sampled)/len(iratios_sampled))

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
#conf=configs['Manhattan3']
conf=configs['Manhattan5']
#conf=configs['Manhattan11']
#conf=configs['CircGraph']
#conf=configs['TKGraph']
conf['direction_north']=False

#SimulatePursuersPathways(conf)
#SimulateInteractiveMode(conf)
SimulateRandomWalker(conf, number_of_runs=1000, print_runs=True, save_plots=False)