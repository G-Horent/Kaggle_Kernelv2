import networkx as nx
import numpy as np
from time import time
from n_walk import product_graph, compute_adj_matrix, get_labels_nodes, get_labels_edges, get_degrees, graph_product
from datetime import datetime
from data import load_training_data, split_data, load_test_data
from tqdm import tqdm


class Kernel():
    def __init__(self):
        pass

    def kernel_eval(self, g1, g2):
        return 0

    def compute_gram_matrix(self, graph_list):
        nb_graphs = len(graph_list)
        K = np.zeros((nb_graphs, nb_graphs))
        for i in tqdm(range(nb_graphs)):
            for j in range(i, nb_graphs):
                k_ij = self.kernel_eval(graph_list[i], graph_list[j])
                K[i, j] = k_ij
                K[j, i] = k_ij
        return K

    def compute_outer_gram(self, graph_list1, graph_list2):
        nb_graphs1, nb_graphs2 = len(graph_list1), len(graph_list2)
        K = np.zeros((nb_graphs1, nb_graphs2))
        for i in tqdm(range(nb_graphs1)):
            for j in range(i, nb_graphs2):
                K[i, j] = self.kernel_eval(graph_list1[i], graph_list2[j])
        return K


class Kernel_nwalk(Kernel):
    def __init__(self, n=3, save_kernel=True):
        super().__init__()
        self.n = n
        self.save_kernel = save_kernel

    def kernel_eval(self, graph_1, graph_2):
        prod_graph = product_graph(graph_1, graph_2)
        if prod_graph.number_of_edges() == 0:
            return 0
        adj_mat = compute_adj_matrix(prod_graph)
        entry = np.sum(np.linalg.matrix_power(adj_mat, self.n))
        return entry

    def compute_gram_matrix(self, graph_list):
        nb_graphs = len(graph_list)
        K = np.zeros((nb_graphs, nb_graphs))
        for i in range(nb_graphs):
            print(f'Computing line {i}')
            start = time()
            for j in range(i, nb_graphs):
                k_ij = self.kernel_eval(graph_list[i], graph_list[j])
                K[i, j] = k_ij
                K[j, i] = k_ij
            end = time()
            print(f'Finished line {i}, total time: {end - start}  avg. time/it: {(end - start) / (nb_graphs - i)}')

        if self.save_kernel:
            now = datetime.now()
            np.save(f'saved/walk_kernel_train_{self.n}_' + now.strftime("%m%d%_h%m%s") + '.npy', K)

        return K

    def compute_outer_gram(self, list_train, list_eval):
        n_train, n_eval = len(list_train), len(list_eval)
        K_cross = np.zeros((n_eval, n_train))
        for i in range(n_eval):
            print(f'Computing line {i}')
            start = time()
            for j in range(n_train):
                K_cross[i, j] = self.kernel_eval(list_eval[i], list_train[j])
            end = time()
            print(f'Finished line {i}, total time: {end - start}  avg. time/it: {(end - start) / n_train}')

        if self.save_kernel:
            now = datetime.now()
            np.save(f'saved/walk_kernel_{self.n}_eval_' + now.strftime("%m%d%_h%m%s") + '.npy', K_cross)

        return K_cross


class Kernel_RBF(Kernel):
    def __init__(self, sigma=2.0):
        super().__init__()
        self.sigma = sigma

    @staticmethod
    def extract_features(graph_list):
        """
        Prend en entrée une liste de graphes de taille N, retourne une matrice N x dim_features
        :param graph_list:
        :return:
        """
        dim_features = 63
        nb_graphs = len(graph_list)
        features_mat = np.zeros((nb_graphs, dim_features))
        for idx, curr_graph in enumerate(tqdm(graph_list)):
            bin_nodes = np.bincount(get_labels_nodes(curr_graph), minlength=50)
            bin_edges = np.bincount(get_labels_edges(curr_graph), minlength=4)
            bin_degrees = np.bincount(get_degrees(curr_graph), minlength=7)

            if nx.is_k_edge_connected(curr_graph, 0):
                diameter = nx.diameter(curr_graph)
            else:
                diameter = 0
            density = nx.density(curr_graph)
            vec = np.hstack((bin_nodes, bin_edges, bin_degrees, diameter, density))
            features_mat[idx, :] = vec

        return features_mat

    def compute_gram_matrix(self, X):
        # TODO
        return 0

    def compute_outer_gram(self, X, Y):
        # TODO
        return 0


class RandomWalkKernel(Kernel):
    def __init__(self, lam, with_lonely_nodes=True):
        """
        Params  :
            - lam : float
                since (I - W) is not invertible, we inverse (I - lam W)
            - with_lonely_nodes : bool
                whether or not we must keep lonely nodes on the product graph
        """
        super().__init__()
        self.lam = lam
        self.with_lonely_nodes = with_lonely_nodes

    def kernel_eval(self, g1, g2):
        g_prod = graph_product(g1, g2, with_lonely_nodes=self.with_lonely_nodes)
        if g_prod.number_of_edges() == 0:
            return 0
        W = nx.adjacency_matrix(g_prod)
        ImW_inv = np.linalg.inv(np.eye(len(g_prod)) - self.lam * W)
        # p_init = np.ones((len(g_prod)))/len(g_prod)
        # k_result = p_init@ImW_inv@np.ones((len(g_prod)))
        k_result = np.sum(ImW_inv) / len(g_prod)
        return k_result


if __name__ == '__main__':
    length_walk = 3
    training_list, training_labels = load_training_data()
    test_list = load_test_data()
    training_split = split_data(training_list, training_labels)

    nb_subset = 0
    train_subset = training_split[nb_subset][0]
    features = Kernel_RBF.extract_features(training_list)
    print(features)
    # test_k = Kernel_nwalk(n=length_walk, save_kernel=True).gram_cross(train_subset, test_list)
    # np.save(f'saved/test/walk_kernel_3_eval_subset_{nb_subset}_.npy', test_k)
