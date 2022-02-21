import networkx as nx

def convert_pkl_to_graphml():
    #G=nx.read_gpickle('datasets/G_nwb/G_test_bb20km_2900nodes.pkl')
    G=nx.read_gpickle('datasets/G_nwb/G_test_DNB_1km_V=809.pkl')
    #G=nx.read_gpickle('../../datasets/G_nwb/G_test_bb1.pkl')

    nodelist=[]
    coord2id={}
    for i,t in enumerate(list(G.nodes())):
        nodelist.append((i,{'latitude':t[0],'longitude':t[1]}))
        coord2id[t]=i

    edgelist=[]
    for j,e in enumerate(list(G.edges())):
        source=e[0]
        target=e[1]
        edgelist.append((coord2id[source],coord2id[target]))

    H=nx.DiGraph()
    H.add_nodes_from(nodelist)
    H.add_edges_from(edgelist)
    nx.write_graphml(H,'datasets/G_nwb/G_converted.graphml')

import matplotlib.pyplot as plt
def plot_raw_G():
    G=nx.read_graphml('dev/nwr_graphs/G_testDNB_edited.graphml')
    G=G.to_undirected()
    #node_text = dict([(c,str(sp.coord2labels[c])) for c in sp.G.nodes])
    # node_text is dict from noded to labels
    # pos is dict from nodeid to coord
    pos={}
    pos_xy={}
    node_text={}
    nodeid_str2idx={}
    for idx,i in enumerate(list(G.nodes(data=True))):
        nodeid_str=i[0]
        info=i[1]
        longit=info['longitude']
        latit=info['latitude']
        x=info['x']
        y=info['y']
        pos[nodeid_str]=(longit,latit)
        pos_xy[nodeid_str]=(x,y)
        node_text[nodeid_str]=nodeid_str
        nodeid_str2idx[nodeid_str]=idx
    
    # Draw according to long lat values
    nx.draw_networkx_edges(G, pos, edge_color='grey', width=1, alpha=1.)#width=1
    #nx.draw_networkx_labels(G,pos, font_size = 8, labels=node_text, font_color='black')#fontsize=8
    V=len(G.nodes)
    colorlist=["grey"]*V
    colorlist[nodeid_str2idx['646']]='#66FF00'
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color=colorlist, edgecolors='black', alpha=1.)#alhpa=.6
    plt.savefig('dev/nwr_graphs/test_longlat.png')
    
    # Draw according to mercator projection by gephi/geo layout
    plt.clf()
    nx.draw_networkx_edges(G, pos_xy, edge_color='grey', width=1, alpha=1.)#width=1
    #nx.draw_networkx_labels(G,pos, font_size = 8, labels=node_text, font_color='black')#fontsize=8
    V=len(G.nodes)
    colorlist=["grey"]*V
    colorlist[nodeid_str2idx['646']]='#66FF00'
    nx.draw_networkx_nodes(G, pos_xy, node_size=10, node_color=colorlist, edgecolors='black', alpha=1.)#alhpa=.6
    plt.savefig('dev/nwr_graphs/test_xy.png')


plot_raw_G()