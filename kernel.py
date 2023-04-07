import numpy as np
import networkx as nx
from data import load_training_data


def compute_adj_matrix(graph):
    nb_nodes = graph.number_of_nodes()
    A = np.zeros((nb_nodes, nb_nodes))
    list_edges = list(graph.edges)
    for u, v in list_edges:
        A[u, v] = 1
        A[v, u] = 1
    return A


def compute_lagrangian_matrix(graph):
    A = compute_adj_matrix(graph)
    vec_D = np.sum(A, axis=0)
    D = np.diag(vec_D)
    L = D - A
    return L


if __name__ == '__main__':
    train_data, train_labels = load_training_data()
    graph_test = train_data[0]
    graph_edges = list(graph_test.edges)
    A_test = compute_adj_matrix(train_data[0])
    print(A_test)
    print(train_data[0].edges)
    L_test = compute_lagrangian_matrix(train_data[0])
    print(L_test)
