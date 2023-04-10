import networkx as nx
import numpy as np
from n_walk import product_graph, compute_adj_matrix, get_labels_nodes, get_labels_edges, get_degrees, graph_product, \
    compute_diameter, check_cycle
from datetime import datetime
from data import load_training_data, split_data, load_test_data
from tqdm import tqdm
from scipy.spatial import distance_matrix
from itertools import product as iter_product
from scipy.sparse.linalg import cg, LinearOperator # conjugate gradient
import os.path as osp


class Kernel:
    def __init__(self, **kwargs):
        self.name = None
        if 'save_path' in kwargs.keys():
            self.save_path = kwargs['save_path']
        else:
            self.save_path = None

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

        if self.save_path is not None:
            now = datetime.now()
            np.save(osp.join(self.save_path, f'{self.name}_' + now.strftime("%m%d_%H%M%S") + '.npy'), K)
        return K

    def compute_outer_gram(self, graph_list1, graph_list2):
        nb_graphs1, nb_graphs2 = len(graph_list1), len(graph_list2)
        K = np.zeros((nb_graphs1, nb_graphs2))
        for i in tqdm(range(nb_graphs1)):
            for j in range(i, nb_graphs2):
                K[i, j] = self.kernel_eval(graph_list1[i], graph_list2[j])
        if self.save_path is not None:
            now = datetime.now()
            np.save(osp.join(self.save_path, f'{self.name}_outer_' + now.strftime("%m%d_%H%M%S") + '.npy'), K)
        return K


class Kernel_nwalk(Kernel):
    def __init__(self, n=3, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.name = 'Kernel_nwalk'

    def kernel_eval(self, graph_1, graph_2):
        prod_graph = product_graph(graph_1, graph_2)
        if prod_graph.number_of_edges() == 0:
            return 0
        adj_mat = compute_adj_matrix(prod_graph)
        entry = np.sum(np.linalg.matrix_power(adj_mat, self.n))
        return entry


class KernelRBF(Kernel):
    def __init__(self, sigma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

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


class RandomWalkKernelNaive(Kernel):
    def __init__(self, lam, norm1 = True, norm2=True, exclude_lonely_nodes=True, save_kernel=False):
        """
        Params  :
            - lam : float
                since (I - W) is not invertible, we inverse (I - lam W)
            - exclude_lonely_nodes : bool
                whether or not we must remove lonely nodes on the product graph
        """
        super().__init__(name='RandomWalkKernelNaive', save_kernel=save_kernel)
        self.lam = lam
        self.norm1 = norm1
        self.norm2 = norm2
        self.exclude_lonely_nodes = exclude_lonely_nodes

    def kernel_eval(self, g1, g2):
        g_prod = graph_product(g1, g2, exclude_lonely_nodes=self.exclude_lonely_nodes)
        if g_prod.number_of_edges() == 0:
            return 0

        A = nx.adjacency_matrix(g_prod).toarray()
        if self.norm1:
            degrees = np.sum(A, axis=1, keepdims=True)
            W = A/np.where(degrees==0, 1, degrees)
        else :
            W = A

        ImW_inv = np.linalg.inv(np.eye(len(g_prod)) - self.lam * W)

        k_result = np.sum(ImW_inv)    # ones.T @ (I - lam W)^(-1) @ ones
        if self.norm2:
            k_result /= len(g_prod)   # 1/n * ones.T @ (I - lam W)^(-1) @ ones
        return k_result


class RandomWalkKernel(Kernel):
    def __init__(self, lam, norm1 = True, norm2=False, exclude_intruding_nodes=True, exclude_lonely_nodes=False, fast=True, max_iter = 100, save_kernel=False):
        """Initialises RandomWalkKernel class.

        Args:
            lam (float): parameter to reduce the spectral radius of the matrix, to avoid inversion of singular matrices.
            norm1 (bool, optional): Whether we normalise the adjacency matrix by the degree. Grakel don't do it but we think its better. Defaults to True.
            norm2 (bool, optional): Whether we normalise the result by the number of nodes in the product graph (like in the slides). Defaults to False.
            exclude_intruding_nodes (bool, optional): Whether we remove nodes corresponding to the product of nodes with different labels.
                Those nodes are automatically considered with the kronecker trick, but we can decide to remove them. Grakel do not remove them. Defaults to True.
            exclude_lonely_nodes (bool, optional): Whether we remove lonely nodes in the product graph. This is an old option, there is no reason to do so. Defaults to False.
            fast (bool, optional): Whether we use Conjugate gradient to avoid matrix inversion and accelerate the computations. The computation for this method is inpired by
                Vishwanathan 2008, and Grakel implementation. Defaults to True.
            max_iter (int, optional): Max iteration in the conjugate gradient.
                IMPORTANT :
                    If norm1 is True, then 100 provides good accuracy, with not much additional time (5ms per graph).
                    If norm1 is False, 100 leads to much additional time (15ms per graph) (we could then consider 50, 10ms per graph).
                Defaults to 100.
            save_kernel (bool, optional): Whether we save the kernel at the end of the computation. Defaults to False.
        """
        super().__init__(name='RandomWalkKernelNaive', save_kernel=save_kernel)
        self.lam = lam
        self.norm1 = norm1
        self.norm2 = norm2
        self.exclude_lonely_nodes = exclude_lonely_nodes
        self.exclude_intruding_nodes = exclude_intruding_nodes
        self.fast = fast
        self.max_iter = max_iter

        if self.fast and self.exclude_lonely_nodes:
            print("Warning, exclude_lonely_nodes=True is not supported with fast=True. Ignored exclude_lonely_nodes=True")

    def filter_graph(self, g):
        """Process a graph to objects needed in kernel computations

        Args:
            g (nx.Graph): graph to process

        Returns:
            tuple(dict, ndarray of dim 1): First element is filt_adj, a dictionnary whose keys are all the possible pairs of labels of g
                and values are adjacency matrices masked with edges corresponding to those pairs. Second element is g_labels, the array
                of the labels of the nodes of g (same length as g).
        """
        g_labels = np.array(get_labels_nodes(g))
        A = nx.adjacency_matrix(g).toarray()
        if self.norm1:
            degrees = np.sum(A, axis=1, keepdims=True)
            A = A/np.where(degrees == 0, 1, degrees) # Markov transition matrix
        filt_adj = {}
        for (l1, l2) in iter_product(g_labels, g_labels):
            A_l1_l2 = (g_labels == l1)[:,None] * A * (g_labels == l2)[None, :] # entries of A corresponding to an edge between a node labeled l1 and a node labeled l2
            filt_adj[(l1, l2)] = A_l1_l2
        return filt_adj, g_labels

    def compute_gram_matrix(self, graph_list):
        processed_graph_list = [self.filter_graph(g) for g in graph_list]
        return super().compute_gram_matrix(processed_graph_list)

    def compute_outer_gram(self, graph_list1, graph_list2):
        processed_graph_list1 = [self.filter_graph(g) for g in graph_list1]
        processed_graph_list2 = [self.filter_graph(g) for g in graph_list2]
        return super().compute_outer_gram(processed_graph_list1, processed_graph_list2)

    def kernel_eval(self, processed_g1, processed_g2):
        """Computes the Rangom walk kernel value of g1 and g2. Warning ! They must be processed

        Args:
            processed_g1 (tuple(dict, ndarray)): object return by self.filter_graph
            processed_g2 (tuple(dict, ndarray)): object return by self.filter_graph
        """
        fa1, gl1 = processed_g1 # filtered adjacency matrix, graph labels
        fa2, gl2 = processed_g2

        common_labels = set(fa1.keys()) & set(fa2.keys())
        len_prod = len(gl1)*len(gl2)

        if not self.fast:
            A_prod = np.zeros((len_prod, len_prod))
            for lalb in common_labels:
                A_prod += np.kron(fa1[lalb], fa2[lalb])
            # A_prod is now the adj matrix of the product graph (with intruding nodes)

            degrees = np.sum(A_prod, axis=1)
            nb_lonely_nodes = np.count_nonzero(degrees == 0) # we will add this quantity at the end
            A_prod = A_prod[np.ix_(degrees !=0, degrees!=0)]

            if self.exclude_lonely_nodes:
                nb_lonely_nodes = 0
            elif self.exclude_intruding_nodes:
                intruding_nodes = gl1[:,None] != gl2[None,:]
                nb_lonely_nodes -= np.count_nonzero(intruding_nodes)

            ImA_inv = np.linalg.inv(np.eye(A_prod.shape[0]) - self.lam * A_prod)
            k_res = np.sum(ImA_inv) + nb_lonely_nodes

            if self.norm2:
                k_res /= (A_prod.shape[0] + nb_lonely_nodes)
            return k_res

        else: # Fast method
            #The computation for this method is inpired by Vishwanathan 2008, and Grakel implementation.

            A_subs = [(fa1[lalb], fa2[lalb]) for lalb in common_labels]

            if self.exclude_intruding_nodes:
                intruding_nodes = gl1[:,None] != gl2[None,:]
                nb_intruding_nodes = np.count_nonzero(intruding_nodes)
            else:
                nb_intruding_nodes = 0

            def lin_op(r):
                # inspired from grakel, but with few corrections
                wR = np.zeros((len(gl1), len(gl2)))
                R = np.reshape(r, (len(gl1), len(gl2)), order='C')
                for A_sub1, A_sub2 in A_subs:
                    wR += np.linalg.multi_dot((A_sub1, R, A_sub2.T)) #could be done faster with sparse matrices ?
                return r - self.lam * wR.flatten(order='C')

            LO = LinearOperator((len_prod, len_prod), matvec=lin_op)
            sol, _ = cg(LO, np.ones(len_prod), tol=1e-6, maxiter=self.max_iter, atol='legacy')

            # A_prod = np.zeros((len_prod, len_prod)) # To delete
            # for lalb in common_labels:
            #     A_prod += np.kron(fa1[lalb], fa2[lalb])
            # M = np.eye(A_prod.shape[0]) - self.lam * A_prod
            # print(np.max(np.abs(1 - lin_op(sol))), np.max(np.abs(1 - M@sol)), np.max(np.abs(sol)))

            k_res =  np.sum(sol) - nb_intruding_nodes
            if self.norm2:
                k_res /= (len_prod - nb_intruding_nodes)
            return k_res







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
