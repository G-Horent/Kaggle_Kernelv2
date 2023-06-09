import numpy as np
import networkx as nx
from data import load_training_data, split_data
from time import time
import itertools


def compute_adj_matrix(graph):
    nb_nodes = graph.number_of_nodes()
    A = np.zeros((nb_nodes, nb_nodes))
    list_edges = list(graph.edges)
    for u, v in list_edges:
        A[u, v] = 1
        A[v, u] = 1
    return A

def get_labels_nodes(graph):
    list_labels_packed = list(nx.get_node_attributes(graph, 'labels').values())
    list_labels = [x[0] for x in list_labels_packed]

    return list_labels


def get_labels_edges(graph):
    list_labels_packed = list(nx.get_edge_attributes(graph, 'labels').values())
    list_labels = [x[0] for x in list_labels_packed]

    return list_labels


def get_degrees(graph):
    list_degrees = [d for n, d in graph.degree()]
    return list_degrees


def compute_diameter(graph):
    all_shortest_paths = list(nx.all_pairs_shortest_path_length(graph))
    list_largest_paths = [np.max(list(dic.values())) for node, dic in all_shortest_paths]

    largest_path = np.max(list_largest_paths)
    return largest_path


def check_cycle(graph):
    try:
        list_edges = nx.find_cycle(graph)
    except nx.exception.NetworkXNoCycle:
        return 0
    else:
        return 1


def product_graph(g1, g2):
    """
    Version Guillaume
    :param g1:
    :param g2:
    :return:
    """
    labels_g1 = np.array(get_labels_nodes(g1))
    labels_g2 = np.array(get_labels_nodes(g2))

    product_nodes = []
    edges = []

    # Create nodes
    for i in range(labels_g1.shape[0]):
        curr_label = labels_g1[i]
        idx = np.nonzero(labels_g2 == curr_label)[0]
        if idx.shape[0] != 0:
            new_nodes = [(i, x.item(0)) for x in np.nditer(idx)]
            product_nodes += new_nodes

    # Add edges
    for curr_node in product_nodes:
        neighbors_g1 = list(nx.neighbors(g1, curr_node[0]))
        neighbors_g2 = list(nx.neighbors(g2, curr_node[1]))
        # neighbors_g1 = list(g1.adj[curr_node[0]].keys())
        # neighbors_g2 = list(g2.adj[curr_node[1]].keys())

        added_edges = [(curr_node, (u, v)) for u, v in itertools.product(neighbors_g1, neighbors_g2) if
                       labels_g1[u] == labels_g2[v]]
        edges += added_edges

    prod_graph = nx.Graph()
    prod_graph.add_nodes_from(product_nodes)
    prod_graph.add_edges_from(edges)

    relabeled_graph = nx.convert_node_labels_to_integers(prod_graph)

    return relabeled_graph


def graph_product(g1, g2, exclude_lonely_nodes=True):
    """
    Version Léo
    :param g1:
    :param g2:
    :param exclude_lonely_nodes:
    :return:
    """
    prod_edges = []
    labels1 = nx.get_node_attributes(g1, "labels")
    labels2 = nx.get_node_attributes(g2, "labels")
    for e1 in g1.edges:
        for e2 in g2.edges:
            u1, v1 = e1
            u2, v2 = e2
            a1 = labels1[u1]
            b1 = labels1[v1]
            a2 = labels2[u2]
            b2 = labels2[v2]
            if (a1 == a2) and (b1 == b2):
                prod_edges.append(((u1, u2), (v1, v2)))
            if (a1 == b2) and (b1 == a2):
                prod_edges.append(((u1, v2), (v1, u2)))

    g_prod = nx.Graph(prod_edges)
    g_prod = nx.convert_node_labels_to_integers(g_prod)

    if not exclude_lonely_nodes:
        prod_vertices = []
        for v1 in g1.nodes:
            for v2 in g2.nodes:
                if labels1[v1] == labels2[v2]:
                    prod_vertices.append(((v1, v2), {"labels": labels1[v1]}))

        g_prod.add_nodes_from(prod_vertices)
    else:
        for v in g_prod.nodes:
            g_prod.nodes[v]['labels'] = labels1[v[0]]

    return g_prod


def graph_product_el(g1, g2, exclude_lonely_nodes=True):
    """Takes into account edges labels"""
    prod_edges = []
    labels1 = nx.get_node_attributes(g1, "labels")
    labels2 = nx.get_node_attributes(g2, "labels")
    for u1, v1, dl1 in g1.edges(data=True):
        for u2, v2, dl2 in g2.edges(data=True):
            if dl1["labels"][0] == dl2["labels"][0]:
                a1 = labels1[u1]
                b1 = labels1[v1]
                a2 = labels2[u2]
                b2 = labels2[v2]
                if (a1 == a2) and (b1 == b2):
                    prod_edges.append(((u1, u2), (v1, v2)))
                if (a1 == b2) and (b1 == a2):
                    prod_edges.append(((u1, v2), (v1, u2)))

    g_prod = nx.Graph(prod_edges)

    if not exclude_lonely_nodes:
        prod_vertices = []
        for v1 in g1.nodes:
            for v2 in g2.nodes:
                if labels1[v1] == labels2[v2]:
                    prod_vertices.append(((v1, v2), {"labels": labels1[v1]}))

        g_prod.add_nodes_from(prod_vertices)
    else:
        for v in g_prod.nodes:
            g_prod.nodes[v]['labels'] = labels1[v[0]]

    return g_prod

