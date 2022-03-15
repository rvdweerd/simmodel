import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle

def convert_IMApkl_to_graphml():
    sourcedir='datasets/G_nwb/1.IMApkl/'
    targetdir='datasets/G_nwb/2.IMAgraphml/'
    filename='G_UTR_1km_V=1191'
    G=nx.read_gpickle(sourcedir+filename+'.pkl') 
    nodelist=[]
    coord2id={}
    for i,t in enumerate(list(G.nodes())):
        nodelist.append((i,{'latitude':t[1],'longitude':t[0]}))
        coord2id[t]=i

    edgelist=[]
    for j,e in enumerate(list(G.edges())):
        source=e[0]
        target=e[1]
        edgelist.append((coord2id[source],coord2id[target]))

    H=nx.DiGraph()
    H.add_nodes_from(nodelist)
    H.add_edges_from(edgelist)
    nx.write_graphml(H,targetdir+filename+'.graphml')

def convert_Gephi_graphml_to_SIM():
    sourcedir='datasets/G_nwb/3.GEPHIgraphml/'
    filename='G_test_ROT_2km_edited_V=2602'
    targetdir='datasets/G_nwb/4.GEPHI_to_SIM/'

    G=nx.read_graphml(sourcedir+filename+'.graphml')
    nodelist_coord=[]
    edgelist_coord=[]
    nodeid_str2coord={}
    nodeid_str2idx={}
    labels={}
    nodes={}
    for idx,i in enumerate(list(G.nodes(data=True))):
        nodeid_str=i[0]
        info=i[1]
        #if 'longitude' not in info: #manually added node without 
        longit=info['longitude']
        latit=info['latitude']
        #longit=info['x']
        #latit=info['y']
        #x=info['x']
        #y=info['y']
        nodeid_str2idx[nodeid_str]=idx
        nodeid_str2coord[nodeid_str]=(longit,latit)
        nodelist_coord.append((longit,latit))
        labels[(longit,latit)]=idx
        nodes[idx]=(longit,latit)
    for idx,e in enumerate(list(G.edges(data=True))):
        source=e[0]
        target=e[1]
        source_coord=nodeid_str2coord[source]
        target_coord=nodeid_str2coord[target]
        edgelist_coord.append((source_coord, target_coord, {'weight':1, 'N_pref':-1}))
    H=nx.Graph() # creates an undirected graph
    H.add_nodes_from(nodelist_coord)
    H.add_edges_from(edgelist_coord)
    #center_node_nr = nodeid_str2idx[center_nodeid_str]
    #center_node_coord = nodelist_coord[center_node_nr]
    #assert nodes[center_node_nr]  ==  center_node_coord
    pos = dict( (n,n) for n in H.nodes() )

    l=0.02;t=0.02;r=0.01;b=0.02
    centernode, centernode_coord, target_nodes, target_nodes_coord = get_center_and_boundary_nodes(nodes, margins=[l,t,r,b])
    plot_center_and_target_nodes(H, pos, centernode, target_nodes, fname=targetdir+filename)
    obj={'G':H,'labels':labels,'pos':pos,'centernode':centernode,
         'centernode_coord':centernode_coord,'target_nodes':target_nodes,'target_nodes_coord':target_nodes_coord}
    out_file = open(targetdir+filename+'.bin', "wb")
    pickle.dump(obj,out_file)
    out_file.close()
    return H, labels, pos, centernode, centernode_coord, target_nodes, target_nodes_coord

def plot_center_and_target_nodes(G, pos, centernode, target_nodes, fname='test'):
    nx.draw_networkx_edges(G, pos, edge_color='grey', width=1, alpha=1.)#width=1
    #nx.draw_networkx_labels(G,pos, font_size = 8, labels=node_text, font_color='black')#fontsize=8
    V=len(G.nodes)
    sizelist=[20]*V
    colorlist=["grey"]*V
    sizelist[centernode]=25
    colorlist[centernode]='#66FF00'

    for i in target_nodes:
        colorlist[i]='red'
    nx.draw_networkx_nodes(G, pos, node_size=sizelist, node_color=colorlist, edgecolors='white', alpha=1.)#alhpa=.6
    plt.savefig(fname+'.png')


def plot_raw_Gephi():
    G=nx.read_graphml('datasets/G_nwb/3.GEPHIgraphml/G_testDNB_edited.graphml')
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
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color=colorlist, edgecolors='white', alpha=1.)#alhpa=.6
    plt.savefig('datasets/G_nwb/3.GEPHIgraphml/test_longlat.png')

def get_center_and_boundary_nodes(nodes={}, margins=[.1,.1,.1,.1]):
    # nodes: dict from nodenr -> (x,y)
    margin_L=margins[0] # left
    margin_T=margins[1] # top 
    margin_R=margins[2] # right
    margin_B=margins[3] # bottom
    # determine cog and min max x and y's:
    xmin = 1e6
    xmax =-1e6
    ymin = 1e6
    ymax =-1e6
    min_distance2 = 1e6
    cog = np.array([coord for _,coord in nodes.items()]).mean(axis=0)
    for i,coord in nodes.items():
        if coord[0]<xmin: xmin=coord[0]
        if coord[0]>xmax: xmax=coord[0]
        if coord[1]<ymin: ymin=coord[1]
        if coord[1]>ymax: ymax=coord[1]
        distance2 = (coord[0]-cog[0])**2 + (coord[1]-cog[1])**2
        if distance2 < min_distance2: 
            min_distance2=distance2
            centernode_nr=i
            centernode_coord=coord
    
    # now loop again to select nodes within margins
    bordernodes_nr=[]
    bordernodes_coord=[]
    width=xmax-xmin
    height=ymax-ymin
    xmarginL=width*margin_L
    xcutoffL=xmin+xmarginL
    xmarginR=width*margin_R
    xcutoffR=xmax-xmarginR
    ymarginT=height*margin_T
    ycutoffT=ymax-ymarginT
    ymarginB=height*margin_B
    ycutoffB=ymin+ymarginB
    for i,coord in nodes.items():
        x=coord[0]
        y=coord[1]
        if x <= xcutoffL or x >= xcutoffR or y >= ycutoffT or y <= ycutoffB:
            bordernodes_nr.append(i)
            bordernodes_coord.append(coord)
    
    return centernode_nr, centernode_coord, bordernodes_nr, bordernodes_coord





    # Draw according to mercator projection by gephi/geo layout
    # plt.clf()
    # nx.draw_networkx_edges(G, pos_xy, edge_color='grey', width=.5, alpha=1.)#width=1
    # #nx.draw_networkx_labels(G,pos, font_size = 8, labels=node_text, font_color='black')#fontsize=8
    # V=len(G.nodes)
    # colorlist=["grey"]*V
    # colorlist[nodeid_str2idx['646']]='#66FF00'
    # nx.draw_networkx_nodes(G, pos_xy, node_size=10, node_color=colorlist, edgecolors='black', alpha=1.)#alhpa=.6
    # plt.savefig('datasets/G_nwb/3.GEPHIgraphml/test_xy.png')

#convert_IMApkl_to_graphml()
#plot_raw_Gephi()
convert_Gephi_graphml_to_SIM()