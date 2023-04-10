import numpy as np
import cvxpy as cp
from data import load_training_data, load_test_data, split_data
from kernel_class import Kernel_nwalk, RandomWalkKernel, KernelRBF
from time import time
from utils import predictions_to_csv


def get_kernel(name, **kwargs):
    if name == 'KernelRBF':
        return KernelRBF(sigma=kwargs['sigma'], **kwargs)
    elif name == 'Kernel_nwalk':
        return Kernel_nwalk(n=kwargs['n'], **kwargs)
    elif name == 'RandomWalkKernel':
        return RandomWalkKernel(lam=kwargs['lam_rand_walk'], **kwargs)
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
    def __init__(self, lmbd=1., kernel_name='KernelRBF', precomputed_kernel=False, kernel_path='', **kwargs):
        super().__init__(kernel_name=kernel_name, **kwargs)
        self.lmbd = lmbd
        # self.kernel = kernel
        self.alpha = None
        self.params = None
        self.X_support = None
        self.X_mean = None
        self.X_std = None
        self.precomputed_kernel = precomputed_kernel
        self.kernel_path = kernel_path
        self.alpha_support = None

    def fit(self, graph_list_train, y):
        N = graph_list_train.shape[0]

        if self.kernel.name == "KernelRBF":
            # If Kernel RBF, compute feature matrix
            print('Extracting features (KernelRBF)')
            X_train = self.kernel.extract_features(graph_list_train)

            # Normalizing features
            self.X_mean = np.mean(X_train, axis=0)
            self.X_std = np.std(X_train, axis=0)
            self.X_std[np.argwhere(self.X_std == 0)] = 1e-8
            X_train = (X_train - self.X_mean) / self.X_std

        else:
            X_train = graph_list_train

        if not self.precomputed_kernel:
            print('Computing Gram Matrix...')
            K = self.kernel.compute_gram_matrix(X_train)

        else:
            K = np.load(self.kernel_path)

        # Ensure K is PSD, else project it to PSD cone w.r.t. Frobenius norm
        eigval, eigvec = np.linalg.eigh(K)
        # Put negative eigenvalues to 0
        eigval[eigval < 0] = 0

        K = eigvec @ np.diag(eigval) @ eigvec.T

        # Sanity check
        assert K.shape[0] == N
        assert K.shape[1] == N

        print("Fitting KernelSVM")
        alpha = cp.Variable(N)
        obj = cp.Maximize(2 * alpha.T @ y - cp.quad_form(alpha, cp.psd_wrap(K)))
        constraints = [0 <= cp.multiply(y, alpha), cp.multiply(y, alpha) <= 1 / (2 * self.lmbd * N)]
        start = time()
        prob = cp.Problem(obj, constraints)
        result = prob.solve(verbose=False)
        end = time()
        print(f'QP Solved in {end - start} secs')
        self.alpha = alpha.value
        print(self.alpha)
        idx_alpha_support = np.nonzero(np.abs(self.alpha) > 10e-8)
        print(idx_alpha_support[0].shape)
        self.alpha_support = np.copy(self.alpha[idx_alpha_support])
        self.X_support = np.copy(X_train[idx_alpha_support])

    def predict(self, graph_list_test):
        if self.kernel.name == 'KernelRBF':
            X_test = self.kernel.extract_features(graph_list_test)
            X_test = (X_test - self.X_mean) / self.X_std

        else:
            X_test = graph_list_test

        print("Computing kernel_outer")
        kernel_outer = self.kernel.compute_outer_gram(X_test, self.X_support)
        return kernel_outer @ self.alpha_support

    def score(self, X, y):
        y_pred = self.predict(X)

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
