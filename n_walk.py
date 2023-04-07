import numpy as np
import networkx as nx
from kernel import compute_adj_matrix
from data import load_training_data, split_data
from time import time
from test import test_product_graph


def kernel_eval(graph_1, graph_2, n):
    prod_graph = test_product_graph(graph_1, graph_2)
    if prod_graph.number_of_edges() == 0:
        return 0
    adj_mat = compute_adj_matrix(prod_graph)
    entry = np.sum(np.linalg.matrix_power(adj_mat, n))
    return entry


def compute_gram_matrix(graph_list, n):
    nb_graphs = len(graph_list)
    K = np.zeros((nb_graphs, nb_graphs))
    for i in range(nb_graphs):
        print(f'Computing line {i}')
        start = time()
        for j in range(i, nb_graphs):
            k_ij = kernel_eval(graph_list[i], graph_list[j], n)
            K[i, j] = k_ij
            K[j, i] = k_ij
        end = time()
        print(f'Finished line {i}, total time: {end - start}  avg. time/it: {(end - start) / (nb_graphs - i)}')
    return K


if __name__ == '__main__':
    length_walk = 3
    training_list, training_labels = load_training_data()
    training_split = split_data(training_list, training_labels)

    test_subset = training_split[0][0]
    test_k = compute_gram_matrix(test_subset, length_walk)
    np.save(f'saved/walk_kernel_{length_walk}.npy', test_k)

    print(test_k)
