import numpy as np
import cvxpy as cp
from data import load_training_data, load_test_data, split_data
from kernel_class import Kernel_nwalk, RandomWalkKernel, KernelRBF
from time import time
from utils import predictions_to_csv


def get_kernel(name, **kwargs):
    if name == 'KernelRBF':
        return KernelRBF(sigma=kwargs['sigma'])
    elif name == 'Kernel_nwalk':
        return Kernel_nwalk(n=kwargs['n'])
    elif name == 'RandomWalkKernel':
        adapted_kwargs = {k: v for (k,v) in kwargs.items() if k!='rwlam'}
        adapted_kwargs['lam'] = kwargs['rwlam']
        return RandomWalkKernel(lam=kwargs['rwlam'])
    else:
        raise NotImplementedError('Unknown kernel')


class KernelMethod:
    def __init__(self, kernel_name, **kwargs):
        self.kernel = get_kernel(kernel_name, **kwargs)

    def fit(self, X, y):
        return 0

    def predict(self, X_test):
        return 0


class KernelRidgeRegression(KernelMethod):
    def __init__(self, lmbd=1.0, kernel_name='n_walk', precomputed=True, **kwargs):
        super().__init__(kernel_name, **kwargs)
        self.lmbd = lmbd
        self.alpha = None
        self.precomputed = precomputed
        if self.precomputed:
            pass
        else:
            self.kernel = Kernel_nwalk(n=3)

    def fit(self, X, y):
        n = y.shape[0]
        K = self.kernel

        self.alpha = np.linalg.inv(K + self.lmbd * n * np.eye(n)) @ y

    def predict(self, X):
        return 0
        # TODO : add prediction

    def score(self, X, y):
        return 0
        # TODO:


class KernelLogisticRegression:
    def __init__(self):
        super().__init__()


class KernelSVM(KernelMethod):
    def __init__(self, lmbd=1., kernel_name='KernelRBF', precomputed_kernel=False, balanced = False, kernel_path='saved/', **kwargs):
        super().__init__(kernel_name=kernel_name, **kwargs)
        self.lmbd = lmbd
        self.balanced = balanced
        # self.kernel = kernel
        self.precomputed_kernel = precomputed_kernel
        self.kernel_path = kernel_path
        # self.params = None #not used

        self.X_support = None
        self.X_mean = None
        self.X_std = None

        self.alpha = None
        self.alpha_support = None

    def fit(self, graph_list_train, y):
        N = graph_list_train.shape[0]

        if self.kernel.name == "KernelRBF":
            # If Kernel RBF, compute feature matrix
            X_train = self.kernel.extract_features(graph_list_train)

            # Normalizing features
            self.X_mean = np.mean(X_train, axis=0)
            self.X_std = np.std(X_train, axis=0)
            self.X_std[np.argwhere(self.X_std == 0)] = 1e-8
            X_train = (X_train - self.X_mean) / self.X_std

        else:
            X_train = graph_list_train

        if not self.precomputed_kernel:
            K = self.kernel.compute_gram_matrix(X_train)
            w, v = np.linalg.eigh(K)

            print(K)
            print(np.linalg.eigh(K))
        else:
            K = np.load(self.kernel_path)
        
        if self.balanced:
            N_pos = np.count_nonzero(y==1)
            weights = np.where(y==1, N/(2*N_pos), N/(2*(N - N_pos)))
        else:
            weights = np.ones((N))

        print("Fitting KernelSVM")
        alpha = cp.Variable(N)
        # obj = cp.Maximize(2 * alpha.T @ y - cp.quad_form(alpha, cp.psd_warp(K)))
        obj = cp.Maximize(2 * alpha.T @ y - cp.quad_form(alpha, K))
        constraints = [0 <= cp.multiply(y, alpha), cp.multiply(y, alpha) <= weights / (2 * self.lmbd * N)]
        start = time()
        prob = cp.Problem(obj, constraints)
        result = prob.solve()
        end = time()
        print(f'QP Solved in {end - start} secs')
        self.alpha = alpha.value
        print(self.alpha)
        idx_alpha_support = np.nonzero(np.abs(self.alpha) > 10e-8)
        print(idx_alpha_support[0].shape)
        self.alpha_support = np.copy(self.alpha[idx_alpha_support])
        self.X_support = np.copy(X_train[idx_alpha_support])

    def predict(self, graph_list_test, precomputed=False, kernel_outer_path = "saved/"):
        if not precomputed:
            if self.kernel.name == 'KernelRBF':
                X_test = self.kernel.extract_features(graph_list_test)
                X_test = (X_test - self.X_mean) / self.X_std

            else:
                X_test = graph_list_test

            print("Computing kernel_outer")
            kernel_outer = self.kernel.compute_outer_gram(X_test, self.X_support)
            return kernel_outer @ self.alpha_support
        
        else:
            kernel_outer = np.load(kernel_outer_path)
            return kernel_outer @ self.alpha

    def score(self, X, y, precomputed=False, kernel_outer_path = "saved/", score_type='accuracy'):
        y_pred = self.predict(X, precomputed=precomputed, kernel_outer_path=kernel_outer_path)

        if score_type == 'AUROC':
            auc = np.count_nonzero(y_pred[y == 1][:,None] > y_pred[y != 1][None, :])
            auc /= np.count_nonzero(y == 1)*np.count_nonzero(y != 1)
            return auc
        else:
            return np.sum(np.sign(y_pred) == y) / y.shape[0]


if __name__ == '__main__':
    train_data, train_labels = load_training_data()
    test_data = load_test_data()

    svm = KernelSVM(lmbd=0.00001, kernel_name='KernelRBF', precomputed_kernel=False, sigma=1.0)

    train_split = split_data()

    svm.fit(train_split[0][0], train_split[0][1])
    predictions = svm.predict(test_data)
    score = svm.score(train_split[1][0], train_split[1][1])
    print(score)
    print(f"Number of negative values: {np.count_nonzero(predictions < 0)}")
    predictions_to_csv('submissions/test_submission.csv', predictions)
