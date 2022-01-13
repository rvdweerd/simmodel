import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import plotly.graph_objects as go
import networkx as nx

def plot_traindata(episode_returns,losses,logdir='./temp'):
    plt.plot(episode_returns)
    plt.savefig(logdir+'/testplots_returns_curve.png')
    plt.clf()
    plt.plot(losses)
    plt.savefig(logdir+'/testplots_loss_curve.png')
    plt.clf()

def PlotAgentsOnGraph_(sp, escape_pos, pursuers_pos, timestep, fig_show=False, fig_save=True, filename=None):
    G=sp.G#.to_directed()
    labels=sp.labels
    pos=sp.pos
    plt.clf()
    colorlist = [1 for _ in range(sp.V)]
    sizelist =  [400 for _ in range(sp.V)]
    node_text = dict([(c,str(sp.coord2labels[c])) for c in sp.G.nodes])
    colorlist=["white"]*sp.V
    colorlist[sp.labels2nodeids[escape_pos]]='#FF0000'
    sizelist[sp.labels2nodeids[escape_pos]]=600
    
    #node_text[sp.labels2coord[escape_pos]]='e'
    for i,P_pos in enumerate(pursuers_pos):
        colorlist[sp.labels2nodeids[P_pos]]='#0000FF'
        sizelist[sp.labels2nodeids[P_pos]]=600
        #fontcolors[sp.labels2nodeids[P_pos]]='white'
        #node_text[sp.labels2coord[P_pos]]='u'+str(i)

    options = {
    "font_color": 'grey',
    "alpha": 1.,
    "font_size": 8,
    "with_labels": False,
    "node_size": 400,
    "node_color": colorlist,#"white",
    "linewidths": .5,
    "labels": node_text,#labels,
    "edge_color": "black",#["black","black","yellow","black","black","black","black"],
    "edgecolors": ["black"]*sp.V,#["black","black","red","black","black","black","black"],
    "width": .5,
    }
    #nx.relabel_nodes(G, labels, copy=True)
    #nx.convert_node_labels_to_integers(G)
    
    #matplotlib.rcParams['figure.figsize'] = [7, 7]
    #nx.draw_networkx(G, pos, **options)
    nx.draw_networkx_edges(G, pos, edge_color='grey', width=1, alpha=1.)
    nx.draw_networkx_labels(G,pos,font_size=8,labels=node_text,font_color='black')
    nx.draw_networkx_nodes(G, pos, node_size=sizelist, node_color=colorlist, alpha=.6)
    #nx.draw_networkx_nodes(G, pos, node_size=10, node_color="k")
    
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = -0.5, wspace = 0)
    ax = plt.gca()
    ax.set_aspect('equal')
    if filename == None:
        pass
        #plt.savefig('test_t='+str(timestep)+'.png')
    else:
        plt.savefig(filename+'.png')
    out=plt.gcf()
    #plt.clf()
    plt.close()
    #plt.figure()

    return out
    

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
        fig.write_image('images_sim/test_t='+str(timestep)+'.png',width=250, height=300,scale=2)
        img=mpimg.imread('images_sim/test_t='+str(timestep)+'.png')
        imgplot=plt.imshow(img)
        plt.show()

def PlotPerformanceCharts(algos,performance_metrics):
    num_iter = performance_metrics['e_returns'][algos[0].__name__].shape[1]
    for algo in algos:
        metrics_mean = np.mean(performance_metrics['e_returns'][algo.__name__], axis=0)
        metrics_std = np.std(performance_metrics['e_returns'][algo.__name__], axis=0)
        xs = np.arange(num_iter)
        ys = metrics_mean
        stds = metrics_std
        lower_std = np.array([val - std for val, std in zip(ys, stds)])
        upper_std = np.array([val + std for val, std in zip(ys, stds)])
        #plt.figure()
        global_min=np.min(performance_metrics['e_returns'][algo.__name__])
        global_max=np.max(performance_metrics['e_returns'][algo.__name__])
        lower_std=np.clip(lower_std,global_min,global_max)
        upper_std=np.clip(upper_std,global_min,global_max)
        plt.plot(xs, ys, label=algo.__name__)
        plt.fill_between(xs, lower_std, upper_std, alpha=0.4)
        plt.legend()
    plt.xlabel("Episode")
    plt.title("Episode returns")
    #plt.ylim((-8,6))
    plt.show()
    plt.savefig('images/rl/test_lcurve.png')


import math
from typing import List
from itertools import chain

# Start and end are lists defining start and end points
# Edge x and y are lists used to construct the graph
# arrowAngle and arrowLength define properties of the arrowhead
# arrowPos is None, 'middle' or 'end' based on where on the edge you want the arrow to appear
# arrowLength is the length of the arrowhead
# arrowAngle is the angle in degrees that the arrowhead makes with the edge
# dotSize is the plotly scatter dot size you are using (used to even out line spacing when you have a mix of edge lengths)
def addEdge(start, end, edge_x, edge_y, lengthFrac=1, arrowPos = None, arrowLength=0.025, arrowAngle = 30, dotSize=20):

    # Get start and end cartesian coordinates
    x0, y0 = start
    x1, y1 = end

    # Incorporate the fraction of this segment covered by a dot into total reduction
    length = math.sqrt( (x1-x0)**2 + (y1-y0)**2 )
    dotSizeConversion = .0565/20 # length units per dot size
    convertedDotDiameter = dotSize * dotSizeConversion
    lengthFracReduction = convertedDotDiameter / length
    lengthFrac = lengthFrac - lengthFracReduction

    # If the line segment should not cover the entire distance, get actual start and end coords
    skipX = (x1-x0)*(1-lengthFrac)
    skipY = (y1-y0)*(1-lengthFrac)
    x0 = x0 + skipX/2
    x1 = x1 - skipX/2
    y0 = y0 + skipY/2
    y1 = y1 - skipY/2

    # Append line corresponding to the edge
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None) # Prevents a line being drawn from end of this edge to start of next edge
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

    # Draw arrow
    if not arrowPos == None:

        # Find the point of the arrow; assume is at end unless told middle
        pointx = x1
        pointy = y1


        if arrowPos == 'middle' or arrowPos == 'mid':
            pointx = x0 + (x1-x0)/2
            pointy = y0 + (y1-y0)/2

        # Find the directions the arrows are pointing
        #signx = -(x1-x0)/abs(x1-x0) if x1!=x0 else -1    #verify this once
        #signy = (y1-y0)/abs(y1-y0) if y1!=y0 else 1    #verified
        eps=1e-7
        PI=math.pi
        if abs(x1-x0)<eps and abs(y1-y0)<eps:
            return
        pointsStraightUp   = abs(x1-x0)<eps and (y1-y0)>0
        pointsStraightDown = abs(x1-x0)<eps and (y1-y0)<0
        pointsStraightRight= x1>x0 and abs(y1-y0)<eps
        pointsStraightLeft = x1<x0 and abs(y1-y0)<eps
        pointsRight = x1>x0
        pointsLeft = x1<x0
        pointsUp = y1>y0
        pointsDown=y1<y0
        if pointsStraightUp:
            phi=90*PI/180
        elif pointsStraightLeft:
            phi=180*PI/180
        elif pointsStraightDown:
            phi=270*PI/180      
        elif pointsStraightRight:              
            phi=0*PI/180
        elif pointsRight and pointsUp:
            phi=math.atan((y1-y0)/(x1-x0))
        elif pointsLeft and pointsUp:
            phi=90*PI/180 - math.atan((y1-y0)/(x1-x0))
        elif pointsLeft and pointsDown:
            phi=180*PI/180 + math.atan((y1-y0)/(x1-x0))        
        elif pointsRight and pointsDown:
            phi=270*PI/180 - math.atan((y1-y0)/(x1-x0))                
        else:
            assert False        
        phi_DEG = math.degrees(phi)
        alpha_DEG = arrowAngle
        alpha = math.radians(arrowAngle)
        delta = phi - alpha

        # Append first arrowhead
        dx = - arrowLength * math.cos(delta)
        dy = - arrowLength * math.sin(delta)
        edge_x.append(pointx)
        edge_x.append(pointx + dx)
        edge_x.append(None)
        edge_y.append(pointy)
        edge_y.append(pointy + dy)
        edge_y.append(None)

        delta = phi + alpha
        dx = - arrowLength * math.cos(delta)
        dy = - arrowLength * math.sin(delta)
        edge_x.append(pointx)
        edge_x.append(pointx + dx)
        edge_x.append(None)
        edge_y.append(pointy)
        edge_y.append(pointy + dy)
        edge_y.append(None)


        # And second arrowhead
        # dx = arrowLength * math.sin(math.radians(eta - arrowAngle))
        # dy = arrowLength * math.cos(math.radians(eta - arrowAngle))
        # edge_x.append(pointx)
        # edge_x.append(pointx + signx**1 * signy * dx)
        # edge_x.append(None)
        # edge_y.append(pointy)
        # edge_y.append(pointy + signx**1 * signy * dy)
        # edge_y.append(None)


    return edge_x, edge_y

def add_arrows(source_x: List[float], target_x: List[float], source_y: List[float], target_y: List[float],
               arrowLength=0.025, arrowAngle=30):
    pointx = list(map(lambda x: x[0] + (x[1] - x[0]) / 2, zip(source_x, target_x)))
    pointy = list(map(lambda x: x[0] + (x[1] - x[0]) / 2, zip(source_y, target_y)))
    etas = list(map(lambda x: math.degrees(math.atan((x[1] - x[0]) / (x[3] - x[2]))),
                    zip(source_x, target_x, source_y, target_y)))

    signx = list(map(lambda x: (x[1] - x[0]) / abs(x[1] - x[0]), zip(source_x, target_x)))
    signy = list(map(lambda x: (x[1] - x[0]) / abs(x[1] - x[0]), zip(source_y, target_y)))

    dx = list(map(lambda x: arrowLength * math.sin(math.radians(x + arrowAngle)), etas))
    dy = list(map(lambda x: arrowLength * math.cos(math.radians(x + arrowAngle)), etas))
    none_spacer = [None for _ in range(len(pointx))]
    arrow_line_x = list(map(lambda x: x[0] - x[1] ** 2 * x[2] * x[3], zip(pointx, signx, signy, dx)))
    arrow_line_y = list(map(lambda x: x[0] - x[1] ** 2 * x[2] * x[3], zip(pointy, signx, signy, dy)))

    arrow_line_1x_coords = list(chain(*zip(pointx, arrow_line_x, none_spacer)))
    arrow_line_1y_coords = list(chain(*zip(pointy, arrow_line_y, none_spacer)))

    dx = list(map(lambda x: arrowLength * math.sin(math.radians(x - arrowAngle)), etas))
    dy = list(map(lambda x: arrowLength * math.cos(math.radians(x - arrowAngle)), etas))
    none_spacer = [None for _ in range(len(pointx))]
    arrow_line_x = list(map(lambda x: x[0] - x[1] ** 2 * x[2] * x[3], zip(pointx, signx, signy, dx)))
    arrow_line_y = list(map(lambda x: x[0] - x[1] ** 2 * x[2] * x[3], zip(pointy, signx, signy, dy)))

    arrow_line_2x_coords = list(chain(*zip(pointx, arrow_line_x, none_spacer)))
    arrow_line_2y_coords = list(chain(*zip(pointy, arrow_line_y, none_spacer)))

    x_arrows = arrow_line_1x_coords + arrow_line_2x_coords
    y_arrows = arrow_line_1y_coords + arrow_line_2y_coords

    return x_arrows, y_arrows

def PlotNodeValues(algos,env,Q_tables):
    for algo in algos:
        G=env.sp.G#.to_directed()
        
        Q_table = Q_tables[algo.__name__]
        V_table={}
        node_values=[0]*env.sp.V
        node_text=['']*env.sp.V
        for s, qvals in Q_table.items():
            V_table[s[0]] = np.max(Q_table[s])
            index = env.sp.labels2nodeids[s[0]]
            node_values[index]=V_table[s[0]]
            node_text[index]='{:.1f}'.format(V_table[s[0]])

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
        # for edge in G.edges():
        #     start=edge[0]
        #     end=edge[1]
        #     edge_x, edge_y = addEdge(start, end, edge_x, edge_y, .8, 'end', .4, 30, 25)
        for s,v in Q_table.items():
            edgeStart = env.sp.labels2coord[s[0]]
            max_actions = np.max(v)
            max_action_indices = np.where(v==max_actions)
            for idx in max_action_indices[0]:
                edgeEnd = env.sp.labels2coord[env.neighbors[s[0]][idx]]
                edge_x, edge_y = addEdge(edgeStart, edgeEnd, edge_x, edge_y, .6, 'end', 0.2, 30, 15)
        #edge_x, edge_y = addEdge((0,0), (1,0), edge_x, edge_y, .6, 'end', 0.2, 30, 15)

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
                colorscale='Greys',#'RdBu',#'Bluered',#'YlGnBu',
                reversescale=True,
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
        node_trace.textfont=dict(family='sans serif',size=5,color='orange')

        fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="title",
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
        fig.write_image('images_rl/test_values.png',width=300, height=300,scale=2)


def PlotGridValues(algos,env,Q_table):
    if env.type == 'graph': return
    cols = env.cols
    rows = env.rows
    vmin = env.final_reward * -1.2
    vmax = env.final_reward

    for algo in algos:
        # Vanilla
        V_table = np.max(Q_table[algo.__name__], axis=1).reshape(rows, cols)
        #plt.imshow(V_table[::-1][:], vmin=vmin, vmax=vmax, cmap='seismic_r')
        plt.imshow(V_table[::-1][:], vmin=vmin, vmax=vmax, cmap='PiYG')
        plt.title("State values:"+algo.__name__)
        #plt.colorbar()
        plt.show()


        # First Vanilla
        plt.xlim((0, cols))
        plt.ylim((0, rows))
        plt.title("Greedy policy:"+algo.__name__)
        for i in range(cols):
            for j in range(rows):
                state_number = j*cols+i
                state_color = env.get_state_color(state_number)
                plt.gca().add_patch(Rectangle((i, j), 1, 1, linewidth=1,
                                                edgecolor='r', facecolor=state_color))
                max_actions,  = np.where(
                    Q_table[algo.__name__][state_number] == np.max(Q_table[algo.__name__][state_number]))
                if state_color in ["white", "blue"]:
                    if 1 in max_actions:
                        plt.arrow(i+0.5, j+0.5, 0.45, 0, width=0.04,
                                    length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
                    if 2 in max_actions:
                        plt.arrow(i+0.5, j+0.5, 0, -0.45, width=0.04,
                                    length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
                    if 3 in max_actions:
                        plt.arrow(i+0.5, j+0.5, -0.45, 0, width=0.04,
                                    length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
                    if 0 in max_actions:
                        plt.arrow(i+0.5, j+0.5, 0, 0.45, width=0.04,
                                    length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
        plt.show()
        #input('key')