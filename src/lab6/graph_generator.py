import networkx as nx
import numpy as np
import os


def with_resistance(graph, resistance_range=(0, 100)):
    a = resistance_range[0]
    b = resistance_range[1]

    for (i, (u, v, w)) in enumerate(graph.edges(data=True)):
        w['sem'] = 0
        w['index'] = i
        w['R'] = np.random.uniform(a, b)

    return graph


def connected_graph(node_number, edge_probability=0.5):
    G = nx.gnp_random_graph(node_number, edge_probability)
    while not nx.is_connected(G):
        G = nx.gnp_random_graph(node_number, edge_probability)
    return with_resistance(G)


def cubic():
    return with_resistance(nx.cubical_graph())


def bridge_graph(node_number, edge_probability=0.5):
    n = node_number//2
    G1 = nx.gnp_random_graph(n, edge_probability)
    while not nx.is_connected(G1):
        G1 = nx.gnp_random_graph(n, edge_probability)
    G2 = nx.gnp_random_graph(n, edge_probability)
    while not nx.is_connected(G2):
        G2 = nx.gnp_random_graph(n, edge_probability)
    mapping = {i: n + i for i in range(n)}
    nx.relabel_nodes(G2, mapping, copy=False)
    G = nx.compose(G1, G2)
    G.add_edge(0, n + 1)
    return with_resistance(G)


def grid(node_number):
    Grid = nx.grid_2d_graph(int(np.sqrt(node_number)), int(np.sqrt(node_number)))
    G = nx.Graph()
    for u, v in Grid.edges:
        G.add_edge(int(np.sqrt(node_number)) * u[0] + u[1], int(np.sqrt(node_number)) * v[0] + v[1])
    return with_resistance(G)


def save_graph_to_file(graph, file_name):
    if os.path.exists(file_name):
        os.remove(file_name)

    text = ""

    for u, v, data in graph.edges(data=True):
        text = text + str(u) + " " + str(v) + " " + str(data["R"]) + "\n"

    filepath = os.path.join(os.getcwd(), file_name)
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(text)


def read_graph_from_file(file_name):
    G = nx.Graph()
    with open(file_name, "r") as file:
        i = 0
        for line in file:
            elements = line.split()
            if len(elements) != 3:
                continue

            u = int(elements[0])
            v = int(elements[1])
            R = float(elements[2])

            G.add_edge(u, v, R=R, sem=0, index=i)
            i += 1
    return G


def gen_and_save(graph_type, file_name, **kwargs):
    save_graph_to_file(graph_type(**kwargs), file_name)
