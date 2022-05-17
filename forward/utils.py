import random

import networkx as nx


def longest_path_algorithm(G):
    """The longest path algorithm is a layer assignement algorithm which returns
    a minimum height layering in linear time. The input DAG, G, is required to
    be a NetworkX DiGraph. The function returns a layering encoded as a list of
    lists. Layer indexes start from 0, sinks are assigned to layer height - 1.
    https://cs.brown.edu/people/rtamassi/gdhandbook/
    """

    # initialize the layer assignement
    assignement = {}

    V = set(G.nodes())
    U = set()
    Z = set()
    current_layer = 1

    while U != V:
        selected = False
        for v in V - U:
            # select a node which has all immediate successors already assigned
            # to Z
            if set(G.successors(v)) <= Z:
                selected = True
                assignement[v] = current_layer
                U.add(v)
                break
        
        if not selected:
            current_layer += 1
            Z |= U
    
    # produce the layering and invert layer IDs
    L = [[] for _ in range(current_layer)]
    for v, l in assignement.items():
        L[current_layer-l].append(v)
    
    return L

def push_sources(G, L):
    """The function pushes the source nodes of the G DAG towards the first layer
    of layering L.
    """

    # retrieve source nodes
    sources = list(filter(lambda x: x[1] == 0, list(G.in_degree())))
    sources = [u for u, _ in sources]
    
    # push sources towards the first layer
    L[0] = sorted(sources)
    for i in range(1, len(L)):
        L[i] = sorted(list(set(L[i]) - set(sources)))
    
    return L

def to_dag(G, seed=0):
    """TODO: add documentation
    """

    node_names = list(G.nodes)
    random.seed(seed)
    random.shuffle(node_names)

    dag = nx.DiGraph()
    for u, v in G.edges:
        u_id = node_names.index(u)
        v_id = node_names.index(v)
        if u_id < v_id:
            dag.add_edge(u_id, v_id)
        if u_id > v_id:
            dag.add_edge(v_id, u_id)
    
    return dag