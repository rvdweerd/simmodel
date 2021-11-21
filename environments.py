import simdata_utils as su
import random 
import time

class GraphWorld(object):
    """"""
    def __init__(self, config):
        self.type='GraphWorld'
        self.sp = su.DefineSimParameters(config)
        dirname = su.make_result_directory(self.sp)
        self.register, self.databank, self.iratios = su.LoadDatafile(dirname)
        self.u_paths=[]
        self.state=()
        self.t=0
        self.reset()
        self.vec2dir={(0,1):'N',(1,0):'E',(0,-1):'S',(-1,0):'W'}
        self.dir2vec={d:c for c,d in self.vec2dir.items()}
        self.reachable_nodes = []

    def _getUpositions(self,t=0):
        upos=[]
        for i,P_path in enumerate(self.u_paths):
            p = P_path[-1] if t >= len(P_path) else P_path[t]
            upos.append(p)
        return upos

    def _availableActionsInCurrentState(self):
        reachable_coords = list(self.sp.G.neighbors(self.state[0]))
        reachable_nodes = [self.sp.coord2labels[c] for c in reachable_coords]
        reachable_directions = None#[self.vec2dir[(n[0]-self.state[0][0],n[1]-self.state[0][1])] for n in reachable_coords]
        return {'coords': reachable_coords, 'node_labels': reachable_nodes, 'directions': reachable_directions}
            
    def reset(self, databank_entry=None):
        # if called with databank_entry, a specific saved initial position is loaded
        if type(databank_entry)==int:
            data_sample=self.databank[databank_entry]
        else:
            data_sample=random.choice(self.databank)
        e_init=data_sample['start_escape_route'] # (x,y)
        u_init=data_sample['start_units'] # [(x0,y0)_1, (x0,y0)_2, ...]
        self.u_paths=data_sample['paths']
        self.t=0
        self.state = tuple([e_init]+u_init)
        return self.state

    def step(self, next_node):
        #if type(next_node) == str:
        #    dir = self.dir2vec[next_node]
        #    next_node = (self.state[0][0]+dir[0],self.state[0][1]+dir[1])
        self.t+=1
        next_node = self.sp.labels2coord[int(next_node)]
        assert next_node in list(self.sp.G.neighbors(self.state[0]))
        new_Upositions = self._getUpositions(self.t) # list
        self.state = tuple([next_node] + new_Upositions)
        reward = +1
        done = False
        if next_node in new_Upositions: # captured
            done=True
            reward += -10
        if next_node[1] == self.sp.most_northern_y: # northern boundary of manhattan graph reached
            done = True
            reward += +10
        return self.state, reward, done, {}

    def render(self):
        su.PlotAgentsOnGraph(self.sp,[self.state[0]],self.u_paths,[self.t],fig_show=False,fig_save=True)
        time.sleep(1)


configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['Manhattan5']
#conf=configs['Manhattan11']
#conf=configs['TKGraph']
conf['direction_north']=False
env=GraphWorld(conf)
s=env.reset()
print('u paths:',env.u_paths,'\nu positions:')
for t in range(5):
    print(env._getUpositions(t))
print('------------')
done=False
R=0
env.render()
while not done:
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