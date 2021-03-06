import networkx as nx

def graph(N):
    """
    Generate manhattan graph
    
    Parameters
    ----------
    N : int
        width of graph.
    Returns
    -------
    G : networkx graph
        graph of size (N,N).
    labels : dict
        {pos: vertex number}.
    pos : dict
        {vertex:pos}.
    """    
    
    G = nx.grid_2d_graph(N,N)
    pos = dict( (n, n) for n in G.nodes() )
    labels = dict( ((i, j), j* N + i) for i, j in G.nodes() ) # 
    G=G.to_directed()    
    #edgelist=[(e,e,{}) for e in G.nodes()]
    #G.add_edges_from(edgelist)
    return G, labels, pos

def CircGraph():
    # Generate example ring road with spokes graph
    # #               __ 0 ___
    # #              1   /\   2
    # #             / \ /  \ / \
    # #            3 - 4 -  5 - 6
    # #             \ / \  / \ /
    # #              7   \/   8
    # #               \_ 9 __/
    # #
    G = nx.DiGraph()
    G.add_nodes_from([
        (3,6),
        (1,5),
        (5,5),
        (0,3),
        (2,3),
        (4,3),
        (6,3),
        (1,1),
        (5,1),
        (3,0),
    ])
    G.add_edges_from([
        ((3,6),(1,5), {'N_pref':0.4}),
        ((3,6),(5,5), {'N_pref':0.4}),
        ((3,6),(2,3), {'N_pref':0.1}),
        ((3,6),(4,3), {'N_pref':0.1}),

        ((1,5),(3,6), {'N_pref':.4}),
        ((1,5),(0,3), {'N_pref':.4}),
        ((1,5),(2,3), {'N_pref':.2}),
        
        ((5,5),(3,6), {'N_pref':0.4}),
        ((5,5),(4,3), {'N_pref':0.2}),
        ((5,5),(6,3), {'N_pref':0.4}),
        
        ((0,3),(1,5), {'N_pref':0.8}),
        ((0,3),(2,3), {'N_pref':0.2}),
        ((0,3),(1,1), {'N_pref':0.}),
        
        ((2,3),(3,6), {'N_pref':0.4}),
        ((2,3),(1,5), {'N_pref':0.4}),
        ((2,3),(0,3), {'N_pref':0.1}),
        ((2,3),(4,3), {'N_pref':0.1}),
        ((2,3),(1,1), {'N_pref':0.}),
        ((2,3),(3,0), {'N_pref':0.}),

        ((4,3),(3,6), {'N_pref':0.4}),
        ((4,3),(5,5), {'N_pref':0.4}),
        ((4,3),(2,3), {'N_pref':0.1}),
        ((4,3),(6,3), {'N_pref':0.1}),
        ((4,3),(3,0), {'N_pref':0. }),
        ((4,3),(5,1), {'N_pref':0. }),

        ((6,3),(5,5), {'N_pref':0.8}),
        ((6,3),(4,3), {'N_pref':0.2}),
        ((6,3),(5,1), {'N_pref':0. }),

        ((1,1),(0,3), {'N_pref':0.5}),
        ((1,1),(2,3), {'N_pref':0.5}),
        ((1,1),(3,0), {'N_pref':0.}),                

        ((5,1),(4,3), {'N_pref':0.5}),
        ((5,1),(6,3), {'N_pref':0.5}),
        ((5,1),(3,0), {'N_pref':0.}),

        ((3,0),(1,1), {'N_pref':0.25}),
        ((3,0),(2,3), {'N_pref':0.25}),
        ((3,0),(4,3), {'N_pref':0.25}),
        ((3,0),(5,1), {'N_pref':0.25}),
    ])
    
    #H=G.to_undirected()
    pos = dict( (n,n) for n in G.nodes() )
    labels = {
        (3,6):0,
        (1,5):1,
        (5,5):2,
        (0,3):3,
        (2,3):4,
        (4,3):5,
        (6,3):6,
        (1,1):7,
        (5,1):8,
        (3,0):9,
    }

    return G, labels, pos

def TKGraph():
    # Generate example graph from TK's thesis
    # #               -- 2 
    # #              /   /\   4
    # #             /   /  \ / 
    # #            0 - 1    3  
    # #                 \  / \ 
    # #                  \/   6
    # #                   5
    ##
    G = nx.DiGraph()
    G.add_nodes_from([
        (0,1),
        (1,1),
        (2,2),
        (3,1),
        (4,2),
        (2,0),
        (4,0),
    ])
    G.add_edges_from([
        ((0,1),(1,1), {'N_pref':1.}),
        
        ((1,1),(2,2), {'N_pref':.9}),
        ((1,1),(2,0), {'N_pref':.1}),
        
        ((2,2),(0,1), {'N_pref':0.5}),
        ((2,2),(3,1), {'N_pref':0.5}),
        
        ((3,1),(4,2), {'N_pref':0.9}),
        ((3,1),(4,0), {'N_pref':0.1}),
        
        ((2,0),(3,1), {'N_pref':1.0}),        
    ])
    
    #H=G.to_undirected()
    pos = dict( (n,n) for n in G.nodes() )
    labels = {
        (0,1):0,
        (1,1):1,
        (2,2):2,
        (3,1):3,
        (4,2):4,
        (2,0):5,
        (4,0):6,
    }

    return G, labels, pos

def MemGraph():
    # Generate example graph from TK's thesis
    # #
    # #            0 - 1    
    # #                  \ 
    # #                   2
    # #                  / 
    # #            3 - 4
    # #                  \ 
    # #                   5 
    # #                  /
    # #            6 - 7 
    # #
    G = nx.DiGraph()
    G.add_nodes_from([
        (0,3),
        (2,3),
        (4,2),
        (0,1),
        (2,1),
        (4,0),
        (0,-1),
        (2,-1),
    ])
    G.add_edges_from([
        ((0,3),(2,3), {'N_pref':1.}),
        ((2,3),(4,2), {'N_pref':1.}),
        ((0,1),(2,1), {'N_pref':1.}),
        ((2,1),(4,2), {'N_pref':1.}),
        ((2,1),(4,0), {'N_pref':1.}),                        
        ((0,-1),(2,-1), {'N_pref':1.}),                        
        ((2,-1),(4,0), {'N_pref':1.}),                        

        ((2,3),(0,3), {'N_pref':1.}),
        ((4,2),(2,3), {'N_pref':1.}),
        ((2,1),(0,1), {'N_pref':1.}),
        ((4,2),(2,1), {'N_pref':1.}),
        ((4,0),(2,1), {'N_pref':1.}),        
        ((2,-1),(0,-1), {'N_pref':1.}),                        
        ((4,0),(2,-1), {'N_pref':1.}),                        
    ])
    
    #H=G.to_undirected()
    pos = dict( (n,n) for n in G.nodes() )
    labels = {
        (0,3):0,
        (2,3):1,
        (4,2):2,
        (0,1):3,
        (2,1):4,
        (4,0):5,
        (0,-1):6,
        (2,-1):7
    }

    return G, labels, pos

def M3test(v=0):
    # 0 - 1 - 2
    # | \     |
    # 3   4   5
    # |       |
    # 6 - 7 - 8
    G = nx.DiGraph()
    G.add_nodes_from([
        (0,2),
        (1,2),
        (2,2),
        (0,1),
        (1,1),
        (2,1),
        (0,0),
        (1,0),
        (2,0)
    ])
    edgelist=[
        ((0,2),(1,2), {'N_pref':1.}),
        ((0,2),(0,1), {'N_pref':1.}),
        ((1,2),(2,2), {'N_pref':1.}),
        ((2,2),(2,1), {'N_pref':1.}),                        
        ((0,1),(0,0), {'N_pref':1.}),                        
        ((2,1),(2,0), {'N_pref':1.}),                     
        ((0,0),(1,0), {'N_pref':1.}),
        ((1,0),(2,0), {'N_pref':1.}),

        ((1,2),(0,2), {'N_pref':1.}),
        ((0,1),(0,2), {'N_pref':1.}),
        ((2,2),(1,2), {'N_pref':1.}),
        ((2,1),(2,2), {'N_pref':1.}),                        
        ((0,0),(0,1), {'N_pref':1.}),                        
        ((2,0),(2,1), {'N_pref':1.}),                     
        ((1,0),(0,0), {'N_pref':1.}),
        ((2,0),(1,0), {'N_pref':1.}),
    ]
    if v==0:
        edgelist.append(((0,1),(1,1), {'N_pref':1.})) # 3-4
        edgelist.append(((1,1),(0,1), {'N_pref':1.}))
    elif v==1:
        edgelist.append(((0,2),(1,1), {'N_pref':1.})) # 0-4
        edgelist.append(((1,1),(0,2), {'N_pref':1.}))
    elif v==2:
        edgelist.append(((0,0),(1,1), {'N_pref':1.})) # 6-4
        edgelist.append(((1,1),(0,0), {'N_pref':1.}))
    elif v==3:
        edgelist.append(((0,1),(1,1), {'N_pref':1.})) # 3-4
        edgelist.append(((1,1),(0,1), {'N_pref':1.}))
        edgelist.append(((0,2),(1,1), {'N_pref':1.})) # 0-4
        edgelist.append(((1,1),(0,2), {'N_pref':1.}))
    elif v==4:
        edgelist.append(((0,1),(1,1), {'N_pref':1.})) # 3-4
        edgelist.append(((1,1),(0,1), {'N_pref':1.}))
        edgelist.append(((0,2),(1,1), {'N_pref':1.})) # 0-4
        edgelist.append(((1,1),(0,2), {'N_pref':1.}))
        edgelist.append(((0,0),(1,1), {'N_pref':1.})) # 6-4
        edgelist.append(((1,1),(0,0), {'N_pref':1.}))
    G.add_edges_from(edgelist)
    
    #H=G.to_undirected()
    pos = dict( (n,n) for n in G.nodes() )
    labels = {
        (0,2):0,
        (1,2):1,
        (2,2):2,
        (0,1):3,
        (1,1):4,
        (2,1):5,
        (0,0):6,
        (1,0):7,
        (2,0):8
    }

    return G, labels, pos


def MemGraphLong():
    # Generate example graph from TK's thesis
    # #
    # #            0 - 1 - 2   
    # #                     \ 
    # #                      3
    # #                     / 
    # #            4 - 5 - 6
    # #                     \ 
    # #                      7  
    # #                     /
    # #            8 - 9 - 10
    # #
    G = nx.DiGraph()
    G.add_nodes_from([
        (-2,3),
        (0,3),
        (2,3),
        (4,2),
        (-2,1),
        (0,1),
        (2,1),
        (4,0),
        (-2,-1),
        (0,-1),
        (2,-1),
    ])
    G.add_edges_from([
        ((-2,3),(0,3), {'N_pref':1.}),
        ((0,3),(2,3), {'N_pref':1.}),
        ((2,3),(4,2), {'N_pref':1.}),
        ((-2,1),(0,1), {'N_pref':1.}),
        ((0,1),(2,1), {'N_pref':1.}),
        ((2,1),(4,2), {'N_pref':1.}),
        ((2,1),(4,0), {'N_pref':1.}),
        ((-2,-1),(0,-1), {'N_pref':1.}),                        
        ((0,-1),(2,-1), {'N_pref':1.}),                        
        ((2,-1),(4,0), {'N_pref':1.}),                        

        ((0,3),(-2,3), {'N_pref':1.}),
        ((2,3),(0,3), {'N_pref':1.}),
        ((4,2),(2,3), {'N_pref':1.}),
        ((0,1),(-2,1), {'N_pref':1.}),
        ((2,1),(0,1), {'N_pref':1.}),
        ((4,2),(2,1), {'N_pref':1.}),
        ((4,0),(2,1), {'N_pref':1.}), 
        ((0,-1),(-2,-1), {'N_pref':1.}),	       
        ((2,-1),(0,-1), {'N_pref':1.}),                        
        ((4,0),(2,-1), {'N_pref':1.}),                        
    ])
    
    #H=G.to_undirected()
    pos = dict( (n,n) for n in G.nodes() )
    labels = {
        (-2,3):0,
        (0,3):1,
        (2,3):2,
        (4,2):3,
        (-2,1):4,
        (0,1):5,
        (2,1):6,
        (4,0):7,
        (-2,-1):8,
        (0,-1):9,
        (2,-1):10
    }

    return G, labels, pos



def SparseManhattanGraph(nside=5):
    G=nx.Graph()
    N=nside**2
    nodes={i:(i%nside,i//nside) for i in range(N)}
    nodelist=[v for k,v in nodes.items()]
    G.add_nodes_from(nodelist)
    edgelist=[
        (0,5,{'N_pref':-1, 'weight':1}),
        (1,2,{'N_pref':-1, 'weight':1}),
        (1,6,{'N_pref':-1, 'weight':1}),
        (2,7,{'N_pref':-1, 'weight':1}),
        (3,4,{'N_pref':-1, 'weight':1}),
        (3,8,{'N_pref':-1, 'weight':1}),
        (5,6,{'N_pref':-1, 'weight':1}),
        (5,10,{'N_pref':-1, 'weight':1}),
        (7,8,{'N_pref':-1, 'weight':1}),
        (7,12,{'N_pref':-1, 'weight':1}),
        (8,13,{'N_pref':-1, 'weight':1}),
        (9,14,{'N_pref':-1, 'weight':1}),
        (10,15,{'N_pref':-1, 'weight':1}),
        (11,12,{'N_pref':-1, 'weight':1}),
        (11,16,{'N_pref':-1, 'weight':1}),
        (13,14,{'N_pref':-1, 'weight':1}),
        (14,19,{'N_pref':-1, 'weight':1}),
        (15,16,{'N_pref':-1, 'weight':1}),
        (15,20,{'N_pref':-1, 'weight':1}),
        (16,21,{'N_pref':-1, 'weight':1}),
        (17,18,{'N_pref':-1, 'weight':1}),
        (17,22,{'N_pref':-1, 'weight':1}),                
        (18,19,{'N_pref':-1, 'weight':1}),
        (18,23,{'N_pref':-1, 'weight':1}),
        (23,24,{'N_pref':-1, 'weight':1}),
    ]
    edgelist_coord=[(nodes[i[0]],nodes[i[1]],i[2]) for i in edgelist]
    G.add_edges_from(edgelist_coord)
    G=G.to_directed()
    pos = dict( (n,n) for n in G.nodes() )
    labels = dict([(v,k) for k,v in nodes.items()])
    return G, labels, pos

def BifurGraph():
    # Generate example graph from TK's thesis
    # #      o - o - o - o - o - o - o - o - o           
    # #    /
    # #  o - o - o - o - o - o - o - o - o
    # #    \
    # #      o - o - o - o - o - o - o - o - o         
    ##
    G = nx.DiGraph()
    nodelist = [
        (0,0),
        (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0),
        (1,1), (2,1), (3,1), (4,1), (5,1), (6,1), (7,1), (8,1), (9,1),
        (1,-1),(2,-1),(3,-1),(4,-1),(5,-1),(6,-1),(7,-1),(8,-1),(9,-1),        
    ]
    G.add_nodes_from(nodelist)

    edgelist=[]
    edgelist += [ ((i,0), (i+1,0), {'N_pref':1.}) for i in range(1,8)] # center line 
    edgelist += [ ((i+1,0), (i,0), {'N_pref':1.}) for i in range(1,8)] # center line reversed
    edgelist += [ ((i,1), (i+1,1), {'N_pref':1.}) for i in range(1,9)] # top line
    edgelist += [ ((i+1,1), (i,1), {'N_pref':1.}) for i in range(1,9)] # top line reversed
    edgelist += [ ((i,-1), (i+1,-1), {'N_pref':1.}) for i in range(1,9)] # bottom line
    edgelist += [ ((i+1,-1), (i,-1), {'N_pref':1.}) for i in range(1,9)] # bottom line reversed
    edgelist += [
        ((0,0),(1,1), {'N_pref':-1}),
        ((1,1),(0,0), {'N_pref':-1}),
        ((0,0),(1,0), {'N_pref':-1}),
        ((1,0),(0,0), {'N_pref':-1}),
        ((0,0),(1,-1), {'N_pref':-1}),
        ((1,-1),(0,0), {'N_pref':-1})]
    assert len(edgelist)==52
    G.add_edges_from(edgelist)
    
    #H=G.to_undirected()
    pos = dict( (n,n) for n in G.nodes() )
    labels = { n:i for i,n in enumerate(nodelist) }
    return G, labels, pos


def MetroGraph():
    # Generate example graph 
    ##
    G = nx.Graph()
    nodes={
        0: (21,18),
        1: (0,0),
        2: (7,0),        
        3: (13,0),
        4: (21,0),
        5: (3,3),
        6: (10,4),
        7: (14,5),
        8: (18,5),
        9: (21,7),
        10: (0,6),
        11: (2,7),
        12: (5.5,7),
        13: (9,8),#(8.5,7.5),
        14: (13.5,7.5),
        15: (0,10),
        16: (4,10),
        17: (10,9),
        18: (14.5,9.5),
        19: (17.5,9),
        20: (21,10.5),
        21: (5,12.5),
        22: (8.5,11),        
        23: (10.5,13),
        24: (13.5,12),
        25: (16.5,14),
        26: (0,16),
        27: (3,16),
        28: (9.5,15.5),
        29: (14.5,16.5),
        30: (3,18),
        31: (16.5,18),
        32: (16,2.5)
    }
    nodelist=[v for k,v in nodes.items()]
    G.add_nodes_from(nodelist)
    edgelist=[
        (0,25,{'N_pref':-1, 'weight':1}),
        (1,5,{'N_pref':-1, 'weight':1}),        
        (2,6,{'N_pref':-1, 'weight':1}),        
        (3,6,{'N_pref':-1, 'weight':1}),        
        (4,32,{'N_pref':-1, 'weight':1}),        
        (5,6,{'N_pref':-1, 'weight':1}),        
        (5,11,{'N_pref':-1, 'weight':1}),        
        (6,7,{'N_pref':-1, 'weight':1}),        
        (6,12,{'N_pref':-1, 'weight':1}),        
        (6,13,{'N_pref':-1, 'weight':1}),        
        (7,8,{'N_pref':-1, 'weight':1}),        
        (7,14,{'N_pref':-1, 'weight':1}),        
        (7,32,{'N_pref':-1, 'weight':1}),        
        (8,9,{'N_pref':-1, 'weight':1}),        
        (8,19,{'N_pref':-1, 'weight':1}),        
        (10,11,{'N_pref':-1, 'weight':1}),        
        (11,12,{'N_pref':-1, 'weight':1}),        
        (12,13,{'N_pref':-1, 'weight':1}),        
        (12,16,{'N_pref':-1, 'weight':1}),        
        (13,17,{'N_pref':-1, 'weight':1}),        
        (14,17,{'N_pref':-1, 'weight':1}),        
        (14,18,{'N_pref':-1, 'weight':1}),        
        (15,16,{'N_pref':-1, 'weight':1}),        
        (16,21,{'N_pref':-1, 'weight':1}),        
        (17,22,{'N_pref':-1, 'weight':1}),        
        (18,19,{'N_pref':-1, 'weight':1}),        
        (18,22,{'N_pref':-1, 'weight':1}),        
        (18,24,{'N_pref':-1, 'weight':1}),        
        (19,20,{'N_pref':-1, 'weight':1}),        
        (19,25,{'N_pref':-1, 'weight':1}),        
        (20,25,{'N_pref':-1, 'weight':1}),        
        (21,22,{'N_pref':-1, 'weight':1}),        
        (21,27,{'N_pref':-1, 'weight':1}),        
        (22,23,{'N_pref':-1, 'weight':1}),        
        (23,24,{'N_pref':-1, 'weight':1}),        
        (23,28,{'N_pref':-1, 'weight':1}),        
        (24,25,{'N_pref':-1, 'weight':1}),        
        (25,29,{'N_pref':-1, 'weight':1}),        
        (26,27,{'N_pref':-1, 'weight':1}),        
        (27,28,{'N_pref':-1, 'weight':1}),        
        (27,30,{'N_pref':-1, 'weight':1}),        
        (28,29,{'N_pref':-1, 'weight':1}),        
        (29,31,{'N_pref':-1, 'weight':1})
    ]
    edgelist_coord=[(nodes[i[0]],nodes[i[1]],i[2]) for i in edgelist]
    G.add_edges_from(edgelist_coord)
    G=G.to_directed()
    pos = dict( (n,n) for n in G.nodes() )
    # some adjustments to improve plotting 
    pos[(10,4)]=(10,3.5)        # node 6
    pos[(14,5)]=(14,3)          # node 7
    pos[(0,6)]=(0,4)            # node 10
    pos[(5.5,7)]=(5.5,5)        # node 12
    pos[(9,8)]=(7,9)            # node 13
    pos[(13.5,7.5)]=(13.5,6.5)  # node 14
    pos[(10,9)]=(10.5,8)        # node 17
    pos[(3,16)]=(3,15)          # node 27
    pos[(9.5,15.5)]=(9.5,16.5)  # node 28
    pos[(16,2.5)]=(18,1.5)      # node 32
    
    labels = dict([(v,k) for k,v in nodes.items()])

    return G, labels, pos