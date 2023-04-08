import numpy as np
import cvxpy as cp
from kernel_class import Kernel_nwalk, RandomWalkKernel, KernelRBF


class KernelRidgeRegression:
    def __init__(self, lmbd=1.0, kernel='n_walk', precomputed=True, **kwargs):
        super().__init__()
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

    def predict(self, X, y):
        return 0
        # TODO : add prediction

    def score(self, X, y):
        return 0
        # TODO:


class KernelLogisticRegression:
    def __init__(self):
        super().__init__()


class KernelSVM:
    def __init__(self, lmbd=1., kernel=None, precomputed_kernel=None):
        self.lmbd = lmbd
        self.kernel = kernel
        self.alpha = None
        self.params = None

    def fit(self, X, y):
        N_tr = X.shape[0]
        self.X_tr = np.copy(X)

        K = self.kernel_(X, X, **self.params)
        # Define QP and solve it with cvxpy
        alpha = cp.Variable(N_tr)
        objective = cp.Maximize(2 * alpha.T @ y - cp.quad_form(alpha, K))
        constraints = [0 <= cp.multiply(y, alpha), cp.multiply(y, alpha) <= 1 / (2 * self.lmbd * N_tr)]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        self.alpha = alpha.value

        return self
