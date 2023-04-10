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
    def __init__(self, lmbd=1., kernel=None, epsilon=1e-6):
        self.lmbd = lmbd
        self.epsilon = epsilon
        self.kernel = kernel #from kernel class

        self.alpha = None
        self.X_tr = None

        # Support vectors
        self.sv_idx = None

    def fit(self, X, y, K=None):
        if not(X is None):
            N_tr = X.shape[0]
            self.X_tr = np.copy(X)
        else:
            N_tr = K.shape[0]

        if K is None:
            K = self.kernel.compute_gram_matrix(X)

        # Define QP and solve it with cvxpy
        alpha = cp.Variable(N_tr)
        objective = cp.Maximize(2 * alpha.T @ y - cp.quad_form(alpha, K))
        constraints = [0 <= cp.multiply(y, alpha), cp.multiply(y, alpha) <= 1 / (2 * self.lmbd * N_tr)]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        self.alpha = alpha.value

        self.sv_idx = (self.alpha > self.epsilon)

        return self
    
    def predict(self, X_te, outer_K = None):
        """Creates predictions for new data

        Args:
            X_te (ndarray of graphs, dim 1): Data to make predictions on.
            outer_K (ndarray, optional): Precomputed outer gram matrix kernel(X_te, X_tr). Defaults to None, 
                in which case, the outer gram matrix will be computed with the support vectors only.

        Returns:
            ndarray, dim 1: predictions. Those are reals (not between 0 and 1).
        """
        if outer_K is None:
            outer_K = self.kernel.compute_outer_gram(X_te, self.X_tr[self.sv_idx])
            logits = outer_K@self.alpha[self.sv_idx]
        else:
            logits = outer_K@self.alpha
        return logits
        

