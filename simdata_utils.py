#from sim_visualization import plot_results
#from dataclasses import dataclass
#import time
#import networkx as nx
#import random
from datetime import datetime
from pathlib import Path
import os
import pickle
from sim_graphs import graph, CircGraph, TKGraph
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class SimParameters(object):
    def __init__(self):
        self.graph_type = None
        self.G = None 
        self.labels = None
        self.pos = None
        self.N = None
        self.coord2nodeid = None
        self.coord2labels = None
        self.labels2coord = None
        self.labels2nodeids = {}
        self.nodeids2labels = {}
        self.nodeid2coord = None
        self.U = None
        self.L = None
        self.R = None
        self.V = None
        self.T = None
        self.direction_north = None
        self.start_escape_route = None
        self.most_northern_y = None
    def __str__(self):
        out = self.graph_type
        out += ', ('+str(self.N)+'x'+str(self.N)+') nodes, ...'
        return out

def GetConfigs():
    configs = {
        "Manhattan3": {
            'graph_type': "Manhattan",
            'N': 3,    # number of nodes along one side
            'U': 2,    # number of pursuer units
            'L': 4,    # Time steps
            'R': 100,  # Number of escape routes sampled 
            'direction_north': True,       # Directional preference of escaper
            'start_escape_route': 'bottom_center' # Initial position of escaper (always bottom center)
        },
        "Manhattan5": {
            'graph_type': "Manhattan",
            'N': 5,    # number of nodes along one side
            'U': 3,    # number of pursuer units
            'L': 6,    # Time steps
            'R': 200,  # Number of escape routes sampled 
            'direction_north': True,       # Directional preference of escaper
            'start_escape_route': 'bottom_center' # Initial position of escaper (always bottom center)
        },
        "Manhattan11": {
            'graph_type': "Manhattan",
            'N': 11,    # number of nodes along one side
            'U': 3,    # number of pursuer units
            'L': 16,    # Time steps
            'R': 1000,  # Number of escape routes sampled 
            'direction_north': True,       # Directional preference of escaper
            'start_escape_route': 'bottom_center' # Initial position of escaper (always bottom center)
        },
        "CircGraph": {
            'graph_type': "CircGraph",
            'N': 10,    # number of nodes along one side
            'U': 2,    # number of pursuer units
            'L': 6,    # Time steps
            'R': 100,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center' # Initial position of escaper (always bottom center)
        },
        "TKGraph": {
            'graph_type': "TKGraph",
            'N': 6,    # number of nodes along one side
            'U': 1,    # number of pursuer units
            'L': 4,    # Time steps
            'R': 10000,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'left' # Initial position of escaper (always bottom center)
        },

    }
    return configs

def DefineSimParameters(config):
    sp = SimParameters()
    sp.graph_type = config['graph_type']
    sp.U = config['U']              # number of pursuer units
    sp.L = config['L']              # Time steps
    sp.R = config['R']              # Number of escape routes sampled 
    if sp.graph_type == 'Manhattan':
        sp.G, sp.labels, sp.pos = graph(config['N'])
        sp.N = config['N']
        sp.V = sp.N**2        # Total number of vertices
        sp.T = sp.L+1         # Total steps in time taken (L + start node)
        sp.direction_north = config['direction_north']
        sp.start_escape_route = (sp.N//2,0) # bottom center of grid
        sp.most_northern_y = max([c[1] for c in sp.G.nodes])
    elif sp.graph_type == 'CircGraph':
        sp.G, sp.labels, sp.pos = CircGraph()#manhattan_graph(N)
        sp.N = 10             # Number of nodes (FIXED)
        sp.V = 10             # Total number of vertices (FIXED)
        sp.T = sp.L+1         # Total steps in time taken (L + start node)
        sp.direction_north = False # (NOT VERY INTERESTING IF TRUE)
        sp.start_escape_route = sp.nodeid2coord[9]
        sp.most_northern_y = max([c[1] for c in sp.G.nodes])
    elif sp.graph_type == 'TKGraph':
        sp.G, sp.labels, sp.pos = TKGraph()#manhattan_graph(N)
        sp.N = 7              # Number of nodes (FIXED)
        sp.V = 7              # Total number of vertices (FIXED)
        sp.T = sp.L+1         # Total steps in time taken (L + start node)
        sp.direction_north = False # (NOT VERY INTERESTING IF TRUE)
        sp.start_escape_route = sp.nodeid2coord[0]
        sp.most_northern_y = max([c[1] for c in sp.G.nodes])
    # Define mappings between node naming conventions
    sp.nodeid2coord = dict( (i, n) for i,n in enumerate(sp.G.nodes()) )
    sp.coord2nodeid = dict( (n, i) for i,n in enumerate(sp.G.nodes()) )
    sp.coord2labels = sp.labels
    sp.labels2coord = dict( (v,k) for k,v in sp.coord2labels.items())
    for coord, nodeid in sp.coord2nodeid.items():
        label = sp.coord2labels[coord]
        sp.labels2nodeids[label]=nodeid
        sp.nodeids2labels[nodeid]=label

    return sp

def make_dirname(sp):
    timestamp = datetime.now()
    dirname = "datasets/" \
        + str(sp.graph_type) + \
        "_N="+ str(sp.N) + \
        "_U="+ str(sp.U) + \
        "_L="+ str(sp.L) + \
        "_R="+ str(sp.R) + \
        "_Ndir="+ str(sp.direction_north)                        
    return dirname

def make_result_directory(sp):
    ######## Create folder for results ########
    dirname = make_dirname(sp)
    # dirname = "results/" + str(config['name'])
    Path(dirname).mkdir(parents=True, exist_ok=True)
    return dirname

def LoadDatafile(dirname):
    all_lengths_fnames = [f for f in os.listdir(dirname) if f.endswith('.pkl')]
    if len(all_lengths_fnames)==0:
        print('No database found.')
        return {},[],[] # We start a new database
    else:
        biggest_dataset_fname = sorted(all_lengths_fnames, key=lambda s: float(s.split('.pkl')[0].split('_')[-1]))[-1]
        in_file = open(dirname + "/" + biggest_dataset_fname, "rb")
        results = pickle.load(in_file)
        in_file.close()
        print('Database found, contains',len(results['databank']),'entries.',in_file.name)
        return results['register'], results['databank'], results['interception_ratios']

def PlotAgentsOnGraph(sp, escape_pos, pursuers_pos, timestep, fig_show=False, fig_save=True):
    # G: nx graph
    # escape_path:   list of escaper coordinates over time-steps
    # pursuers_path: list list of pursuer coordinates over time-steps
    # timesteps:     list of time-steps to plot
    G=sp.G
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = edge[0]
        x1, y1 = edge[1]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
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
            #showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            #colorscale='YlGnBu',
            #reversescale=True,
            color='#5fa023',#[],
            size=10,
            #colorbar=dict(
            #    thickness=15,
            #    title='Node Connections',
            #    xanchor='left',
            #    titleside='right'
            #),
            line_width=2))

    #node_adjacencies = []
    #for node, adjacencies in enumerate(G.adjacency()):
    #    node_adjacencies.append(len(adjacencies[1]))
    #    node_text.append('a') # of connections: '+str(len(adjacencies[1])))
    colorlist = [1 for _ in range(sp.V)]
    sizelist =  [1 for _ in range(sp.V)]
    node_text = [str(sp.coord2labels[c]) for c in sp.G.nodes]
    colorlist[sp.labels2nodeids[escape_pos]]='#FF0000'
    sizelist[sp.labels2nodeids[escape_pos]]=20
    node_text[sp.labels2nodeids[escape_pos]]='e'
    for i,P_pos in enumerate(pursuers_pos):
        colorlist[sp.labels2nodeids[P_pos]]='#0000FF'
        sizelist[sp.labels2nodeids[P_pos]]=20
        node_text[sp.labels2nodeids[P_pos]]='u'+str(i)
    node_trace.marker.color = colorlist
    node_trace.marker.size = sizelist
    node_trace.text = node_text
    node_trace.textfont = {
            "size": [6 for i in range(sp.V)],
            "color": ['black' for i in range(sp.V)]
        }
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="t="+str(timestep),#'<br>Network graph made with Python',
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
    if fig_show:
        fig.show()
    if fig_save:
        fig.write_image('images/test_t='+str(timestep)+'.png',width=250, height=300,scale=2)
        img=mpimg.imread('images/test_t='+str(timestep)+'.png')
        imgplot=plt.imshow(img)
        plt.show()