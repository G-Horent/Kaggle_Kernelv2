import numpy as np
import cvxpy as cp
from cvxopt import matrix, solvers
from data import load_training_data, load_test_data, split_data
from kernel_class import Kernel_nwalk, RandomWalkKernel, KernelRBF, KernelWLSubtree
from time import time
from utils import predictions_to_csv


def get_kernel(name, **kwargs):
    if name == 'KernelRBF':
        return KernelRBF(sigma=kwargs['sigma'])
    elif name == 'Kernel_nwalk':
        return Kernel_nwalk(n=kwargs['n'])
    elif name == 'RandomWalkKernel':
        adapted_kwargs = {k: v for (k, v) in kwargs.items() if k != 'rwlam'}
        adapted_kwargs['lam'] = kwargs['rwlam']
        return RandomWalkKernel(lam=kwargs['rwlam'])
    elif name == 'KernelWLSubtree':
        return KernelWLSubtree(h=kwargs['h'])
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
<<<<<<< HEAD
    def __init__(self, lmbd=1., kernel_name='KernelRBF', precomputed_kernel=False, balanced=False, kernel_path='saved/',
                 **kwargs):
=======
    def __init__(self, lmbd=1., balanced = False, precomputed_kernel=False, kernel_path='saved/', kernel_name='KernelRBF', **kwargs):
>>>>>>> 64fff42c22df3703069dc279c649e991f6e54982
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
            print('Computing Kernel Gram Matrix...')
            K = self.kernel.compute_gram_matrix(X_train)

        else:
            K = np.load(self.kernel_path)

        w, v = np.linalg.eigh(K)
        w[w < 0] = 0
        K = v @ np.diag(w) @ v.T



        if self.balanced:
            N_pos = np.count_nonzero(y == 1)
            weights = np.where(y == 1, N / (2 * N_pos), N / (2 * (N - N_pos)))
        else:
            weights = np.ones((N))

        print("Fitting KernelSVM")
        alpha = cp.Variable(N)
<<<<<<< HEAD
        # obj = cp.Maximize(2 * alpha.T @ y - cp.quad_form(alpha, cp.psd_warp(K)))
        obj = cp.Maximize(2 * alpha.T @ y - cp.quad_form(alpha, cp.psd_wrap(K)))
=======
        obj = cp.Maximize(2 * alpha.T @ y - cp.quad_form(alpha, cp.psd_warp(K)))
>>>>>>> 64fff42c22df3703069dc279c649e991f6e54982
        constraints = [0 <= cp.multiply(y, alpha), cp.multiply(y, alpha) <= weights / (2 * self.lmbd * N)]
        start = time()
        prob = cp.Problem(obj, constraints)
        result = prob.solve()
        end = time()
        print(f'QP Solved in {end - start} secs')
        self.alpha = alpha.value
        # print(self.alpha)
        idx_alpha_support = np.nonzero(np.abs(self.alpha) > 10e-8)
        print(idx_alpha_support[0].shape)
        self.alpha_support = np.copy(self.alpha[idx_alpha_support])
        self.X_support = np.copy(X_train[idx_alpha_support])

    def predict(self, graph_list_test, precomputed=False, kernel_outer_path="saved/"):
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

    def score(self, X, y, precomputed=False, kernel_outer_path="saved/", score_type='accuracy'):
        y_pred = self.predict(X, precomputed=precomputed, kernel_outer_path=kernel_outer_path)

        if score_type == 'AUROC':
            auc = np.count_nonzero(y_pred[y == 1][:, None] > y_pred[y != 1][None, :])
            auc /= np.count_nonzero(y == 1) * np.count_nonzero(y != 1)
            return auc
        else:
            return np.sum(np.sign(y_pred) == y) / y.shape[0]


class KernelSVM2():
    def __init__(self, lmbd=1., balanced = False, kernel=None):
        """
        Args:
            lmbd (float, optional): Penalisation parameter. Defaults to 1..
            balanced (bool, optional): Whether we must weight samples inversely proportionally to the size of their class. Defaults to False.
            kernel (Kernel, optional): Kernel object to use. Defaults to None.
        """
        self.lmbd = lmbd
        self.balanced = balanced
        self.kernel = kernel

        self.X_support = None
        self.X_mean = None
        self.X_std = None

        self.alpha = None
        self.alpha_support = None
    
    def optimise(self, K, y, c, method="cvxopt"):
        N = K.shape[0]
        if method == "cvxpy":
            print("Fitting KernelSVM")
            alpha = cp.Variable(N)
            obj = cp.Maximize(2 * alpha.T @ y - cp.quad_form(alpha, K))
            constraints = [0 <= cp.multiply(y, alpha), cp.multiply(y, alpha) <= c]

            start = time()
            prob = cp.Problem(obj, constraints)
            result = prob.solve()
            end = time()

            print(f'QP Solved in {end - start} secs')
            return alpha.value
        
        else:
            # min 1/2*(x^T)Qx + (p^T)x     st Gx <= h and Ax = b
            P = matrix(K)
            q = - matrix(y.astype(float))
            G = matrix(np.concatenate([-np.diag(y), np.diag(y)], axis=0).astype(float))
            h = matrix(np.concatenate([np.zeros(N), c]))
            A = matrix(np.ones((1,N))) # if we want to take the version with b
            # A = matrix(np.zeros((1,N)))
            b = matrix(0.)
            sol = solvers.qp(P, q, G, h, A, b)
            return np.array(sol["x"]).flatten()
    
    def fit(self, K, y, X_train=None):
        """Fit the SVM to the corresponding kernel matrix and labels

        Args:
            K (square ndarray): Kernel matrix
            y (ndarray of dim 1): labels (-1 or 1)
            X_train (ndarray of graphs, optional): Corresponding graphs. Not used except to save in an attribute, for future predictions. Defaults to None.
        """
        N = K.shape[0]
        
        if self.balanced:
            N_pos = np.count_nonzero(y==1)
            weights = np.where(y==1, N/(2*N_pos), N/(2*(N - N_pos)))
        else:
            weights = np.ones((N))

        c = weights / (2 * self.lmbd * N)

        self.alpha = self.optimise(K, y, c)
        idx_alpha_support = np.nonzero(np.abs(self.alpha) > 10e-8)[0]
        self.alpha_support = np.copy(self.alpha[idx_alpha_support])
        if X_train is not None:
            self.X_support = np.copy(X_train[idx_alpha_support])

        idx_alpha_margin = np.nonzero((np.abs(self.alpha) > 1e-7) & (np.abs(self.alpha) < c - 1e-7))[0]
        print(len(idx_alpha_support))
        print(len(idx_alpha_margin))

        b_candi = y - K[:,idx_alpha_support]@self.alpha_support # we could have said y - K@self.alpha, but implies errors
        if len(idx_alpha_margin) > 0:
            self.b = np.median(b_candi[idx_alpha_margin]) # offset of the classifier
        else:
            self.b = 0
    
    def fit_saved(self, kernel_path, y, X_train=None):
        K = np.load(kernel_path)
        return self.fit(K, y, X_train=X_train)
    
    def fit_compute(self, graph_list_train, y):
        if self.kernel is None:
            raise AttributeError("fit_compute needs attribute kernel to be difined")
        
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
        K = self.kernel.compute_gram_matrix(X_train)
        return self.fit(K, y, X_train=X_train)
    
    def predict(self, K_outer):
        if self.alpha is None:
            raise AttributeError("The model has to be fitted first before predictions")
        return K_outer @ self.alpha + self.b
    
    def predict_saved(self, outer_kernel_path = "saved/"):
        K_outer = np.load(outer_kernel_path)
        return self.predict(K_outer)
    
    def predict_compute(self, graph_list_test):
        if self.kernel is None:
            raise AttributeError("predict_compute needs attribute kernel to be difined.")
        if self.X_support is None:
            raise AttributeError("predict_compute needs X_train to have been filled in previously in the fit operation.")
        if self.alpha_support is None:
            raise AttributeError("The model has to be fitted first before predictions")
        
        if self.kernel.name == 'KernelRBF':
            X_test = self.kernel.extract_features(graph_list_test)
            X_test = (X_test - self.X_mean) / self.X_std

        else:
            X_test = graph_list_test

        print("Computing kernel_outer")
        kernel_outer = self.kernel.compute_outer_gram(X_test, self.X_support)
        return kernel_outer @ self.alpha_support + self.b

    def _score(self, y_true, y_pred, score_type='accuracy'):

        if score_type == 'AUROC':
            auc = np.count_nonzero(y_pred[y_true == 1][:,None] > y_pred[y_true != 1][None, :])
            auc /= np.count_nonzero(y_true == 1)*np.count_nonzero(y_true != 1)
            return auc
        else:
            return np.sum(np.sign(y_pred) == y_true) / y_true.shape[0]
    
    def score(self, K_outer, y_true, score_type='accuracy'):
        y_pred = self.predict(K_outer)
        return self._score(y_true, y_pred, score_type=score_type)
    
    def score_saved(self, outer_kernel_path, y_true, score_type='accuracy'):
        y_pred = self.predict_saved(outer_kernel_path)
        return self._score(y_true, y_pred, score_type=score_type)
    
    def score_compute(self, graph_list_test, y_true, score_type='accuracy'):
        y_pred = self.predict_compute(graph_list_test)
        return self._score(y_true, y_pred, score_type=score_type)
    
    





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
