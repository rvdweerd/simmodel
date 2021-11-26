import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import matplotlib
import plotly.graph_objects as go
import networkx as nx

def PlotAgentsOnGraph_(sp, escape_pos, pursuers_pos, timestep, fig_show=False, fig_save=True):
    G=sp.G#.to_directed()
    labels=sp.labels
    pos=sp.pos

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
    plt.savefig('test_t='+str(timestep)+'.png')
    #plt.clf()
    

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
    plt.savefig('images_rl/test_lcurve.png')


from addEdge import addEdge

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