from modules.rl.rl_policy import EpsilonGreedyPolicySB3_PPO, Policy
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.rl_utils import EvaluatePolicy, print_parameters, GetFullCoverageSample, NpWrapper
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path
def PlotNodeValues(env,ValueVec,fname='./results/_Densities/test.png',cmax=1.,tstep=0):
    G=env.sp.G#.to_directed()
    V_table={}
    node_values=[0]*env.sp.V
    node_text=['']*env.sp.V
    for s, val in enumerate(ValueVec):
        V_table[s] = val
        index = env.sp.labels2nodeids[s]
        node_values[index]=V_table[s]
        node_text[index]='{:.2f}'.format(V_table[s])

    edge_x = []
    edge_y = []
    # Plot all edges
    for edge in G.edges():
        x0, y0 = edge[0]
        x1, y1 = edge[1]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    # for s,v in Q_table.items():
    #     edgeStart = env.sp.labels2coord[s[0]]
    #     max_actions = np.max(v)
    #     max_action_indices = np.where(v==max_actions)
    #     for idx in max_action_indices[0]:
    #         edgeEnd = env.sp.labels2coord[env.neighbors[s[0]][idx]]
    #         edge_x, edge_y = addEdge(edgeStart, edgeEnd, edge_x, edge_y, .6, 'end', 0.2, 30, 15)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='black'),#'#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = node
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        #hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',#'RdBu',#'Bluered',#'YlGnBu',
            cmin=0.,
            cmax=cmax,
            reversescale=False,
            color=[],#='#5fa023',#[],
            size=15,
            colorbar=dict(
                thickness=5,
                #title='Node values',
                xanchor='left',
                titleside='right' ),
            line_width=.5,
            )
        )

    node_trace.marker.color=node_values
    node_trace.text = node_text
    node_trace.textfont=dict(family='sans serif',size=6,color='orange')

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Probability of Pursuit Unit presence per node, t="+str(tstep),
                    titlefont_size=12,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    # annotations=[ dict(
                    #    text="a",#"Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    #    showarrow=False,
                    #    xref="paper", yref="paper",
                    #    x=0.005, y=-0.002 ) ],
                    # annotations=[ dict(
                    #     x=positions[adjacencies[0]][0],
                    #     y=positions[adjacencies[0]][1],
                    #     text=adjacencies[0], # node name that will be displayed
                    #     xanchor='left',
                    #     xshift=10,
                    #     font=dict(color='black', size=10),
                    #     showarrow=False, arrowhead=1, ax=-10, ay=-10)],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.write_image(fname,width=300, height=300,scale=2)

#world_name='Manhattan3x3_PauseFreezeWorld'
#world_name='Manhattan5x5_FixedEscapeInit'
world_name='MetroU3_e17tborder_FixedEscapeInit'
state_repr='et'
env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc='nodes')
num_worlds=len(env.world_pool)
Path('./results/_Densities/'+world_name).mkdir(parents=True, exist_ok=True)

# Count U presence on each node per timestep
fieldcounts=np.zeros((7,env.sp.V))
for step in range(7):
    for instance in env.databank['labels']:
        paths=instance['paths']
        for p in paths:
            if len(p)<=step: # unit has reached end node, unit path finished (take last entry)
                fieldcounts[step,p[-1]]+=1
            else: # unit path is not finished yet
                fieldcounts[step,p[step]]+=1

# fieldcount=np.zeros(env.sp.V)
# maxlen=-1
# for instance in env.databank['labels']:
#     paths=instance['paths']
#     for p in paths:
#         fieldcount[p[-1]]+=1
#         maxlen=max(maxlen,len(p))

def Plot2dGridDensity(fieldcount, fname='./results/_Densities/test.png'):
    mat = fieldcount.reshape(5,5)[::-1][:]
    labels = fieldcount.reshape(5,5)[::-1][:]/np.sum(fieldcount)
    labs = np.array([['{:.2f}'.format(i) for i in rows] for rows in fieldcount.reshape(5,5)[::-1][:]/np.sum(fieldcount)])
    ax = sns.heatmap(mat, cmap="YlGnBu", linewidths=.5,annot=labs, annot_kws={'fontsize': 10}, fmt='s')
    plt.show()
    plt.savefig(fname)
    plt.clf()
    plt.savefig("./results/_Densities/fieldprobs_check.png")

cmax=np.max(fieldcounts/num_worlds)#(env.sp.U*num_worlds))
for i,f in enumerate(fieldcounts):
    PlotNodeValues(env,f/num_worlds,'./results/_Densities/'+world_name+'/fieldprobs_t='+str(i)+'.png',cmax=cmax,tstep=i)

# Test escape ratios for customized policies
class WaitAndRunPolicy(Policy):
    def __init__(self, env):
        #self.env=env
        super().__init__('wait and run mindeg')
        self.out_degree=env.out_degree
        self.__name__ = 'WaitAndRun'
        self.count=0
        #self.action_seq=[2,2,2,3,3,3]
        self.action_seq=[1,0,0,2,3,3,3]
    def reset_hidden_states(self):
        self.count=-1
    def sample_action(self, s, available_actions=None):
        self.count+=1
        return self.action_seq[self.count], None
#policy=WaitAndRunPolicy(env)
#lengths, returns, captures = EvaluatePolicy(env, policy, env.world_pool, print_runs=False, save_plots=False, logdir='./test')
