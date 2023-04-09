import networkx as nx
import numpy as np
from n_walk import product_graph, compute_adj_matrix, get_labels_nodes, get_labels_edges, get_degrees, graph_product, \
    compute_diameter, check_cycle
from datetime import datetime
from data import load_training_data, split_data, load_test_data
from tqdm import tqdm
from scipy.spatial import distance_matrix


class Kernel:
    def __init__(self):
        self.name = None
        self.save_kernel = True

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

        if self.save_kernel:
            now = datetime.now()
            np.save(f'saved/{self.name}_' + now.strftime("%m%d%_h%m%s") + '.npy', K)
        return K

    def compute_outer_gram(self, graph_list1, graph_list2):
        nb_graphs1, nb_graphs2 = len(graph_list1), len(graph_list2)
        K = np.zeros((nb_graphs1, nb_graphs2))
        for i in tqdm(range(nb_graphs1)):
            for j in range(i, nb_graphs2):
                K[i, j] = self.kernel_eval(graph_list1[i], graph_list2[j])
        if self.save_kernel:
            now = datetime.now()
            np.save(f'saved/{self.name}_outer_' + now.strftime("%m%d%_h%m%s") + '.npy', K)
        return K


class Kernel_nwalk(Kernel):
    def __init__(self, n=3, save_kernel=True):
        super().__init__()
        self.n = n
        self.save_kernel = save_kernel
        self.name = 'Kernel_nwalk'

    def kernel_eval(self, graph_1, graph_2):
        prod_graph = product_graph(graph_1, graph_2)
        if prod_graph.number_of_edges() == 0:
            return 0
        adj_mat = compute_adj_matrix(prod_graph)
        entry = np.sum(np.linalg.matrix_power(adj_mat, self.n))
        return entry


class KernelRBF(Kernel):
    def __init__(self, sigma=2.0):
        super().__init__()
        self.sigma = sigma
        self.name = 'KernelRBF'

    @staticmethod
    def extract_features(graph_list):
        """
        Prend en entrée une liste de graphes de taille N, retourne une matrice N x dim_features
        :param graph_list:
        :return:
        """
        dim_features = 65
        nb_graphs = len(graph_list)
        features_mat = np.zeros((nb_graphs, dim_features))
        for idx, curr_graph in enumerate(tqdm(graph_list)):
            bin_nodes = np.bincount(get_labels_nodes(curr_graph), minlength=50)
            bin_edges = np.bincount(get_labels_edges(curr_graph), minlength=4)
            bin_degrees = np.bincount(get_degrees(curr_graph), minlength=7)
            diameter = compute_diameter(curr_graph)
            density = nx.density(curr_graph)
            has_cycle = check_cycle(curr_graph)
            nb_components = nx.number_connected_components(curr_graph)
            vec = np.hstack((bin_nodes, bin_edges, bin_degrees, diameter, density, has_cycle, nb_components))
            features_mat[idx, :] = vec

        return features_mat

    def kernel_eval(self, g1, g2):
        feat = self.extract_features([g1, g2])
        return np.exp(-(np.linalg.norm(feat[0, :] - feat[1, :])**2)/(2 * self.sigma**2))

    def compute_gram_matrix(self, X):
        return np.exp(-(distance_matrix(X, X)**2)/(2 * (self.sigma**2)))

    def compute_outer_gram(self, X1, X2):
        return np.exp(-(distance_matrix(X1, X2)**2)/(2 * (self.sigma**2)))


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
        self.name = 'RandomWalkKernel'

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
    training_split = split_data()

    nb_subset = 0
    train_subset = training_split[nb_subset][0]
    kernel_class = KernelRBF(sigma=2.0)
    features = kernel_class.extract_features(training_list)
    gram = kernel_class.compute_gram_matrix(features)
    print(gram.shape)
    print(np.linalg.eigh(gram))
    # test_k = Kernel_nwalk(n=length_walk, save_kernel=True).gram_cross(train_subset, test_list)
    # np.save(f'saved/test/walk_kernel_3_eval_subset_{nb_subset}_.npy', test_k)
