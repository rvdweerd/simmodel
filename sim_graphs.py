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
    # #                  \/   5
    # #                   6
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