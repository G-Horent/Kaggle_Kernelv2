import networkx as nx
import numpy as np
from data import load_training_data
import itertools


def test_product_graph(g1, g2):
    labels_g1 = np.array(get_labels(g1))
    labels_g2 = np.array(get_labels(g2))

    product_nodes = []
    edges = []

    # Create nodes
    for i in range(labels_g1.shape[0]):
        curr_label = labels_g1[i]
        idx = np.nonzero(labels_g2 == curr_label)[0]
        if idx.shape[0] != 0:
            new_nodes = [(i, x.item(0)) for x in np.nditer(idx)]
            product_nodes += new_nodes

    product_nodes_set = set(product_nodes)

    for curr_node in product_nodes:
        neighbors_g1 = list(g1.adj[curr_node[0]].keys())
        neighbors_g2 = list(g2.adj[curr_node[1]].keys())

        candidate_nodes = [(u, v) for u, v in itertools.product(neighbors_g1, neighbors_g2) if
                           labels_g1[u] == labels_g2[v]]
        for cand_node in candidate_nodes:
            if cand_node in product_nodes_set:
                edges.append((curr_node, cand_node))

    prod_graph = nx.Graph()
    prod_graph.add_nodes_from(product_nodes)
    prod_graph.add_edges_from(edges)

    relabeled_graph = nx.convert_node_labels_to_integers(prod_graph)

    return relabeled_graph


def get_labels(graph):
    list_labels = []
    list_nodes = list(graph.nodes)
    for node_nb in list_nodes:
        curr_node = graph.nodes[node_nb]['labels']
        list_labels = list_labels + curr_node
    return list_labels


if __name__ == '__main__':
    train_list, train_labels = load_training_data()
    final_graph = test_product_graph(train_list[0], train_list[1])
    print(final_graph.number_of_nodes())
    print(final_graph.number_of_edges())
    print(final_graph.nodes)
    print(final_graph.edges)
