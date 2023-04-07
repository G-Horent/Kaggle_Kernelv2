import numpy as np
import cvxpy as cp


class KernelRidgeRegression:
    def __init__(self, lmbd=1.0, kernel='n_walk'):
        super().__init__()
        self.lmbd = lmbd
        self.kernel = kernel
        self.alpha = None

    def fit(self, X, y):
        n = y.shape[0]
        K = self.kernel(X, X)

        self.alpha = np.linalg.inv(K+self.lmbd*n*np.eye(n))@y

    def predict(self, X, y):
        return 0
        #TODO : add prediction

    def score(self, X, y):
        return 0
        #TODO:






class KernelLogisticRegression:
    def __init__(self):





class KernelSVM:
    def __init__(self, lmbd=1., kernel=None, precomputed_kernel=None):
        self.lmbd = lmbd

        self.kernel = kernel
        self.params = {'lamb': lamb, 'sig': sigma, 'k': k}

    def fit(self, X, y):
        # We keep values of training in memory for prediction
        N_tr = X.shape[0]
        self.X_tr_ = np.copy(X)

        K = self.kernel_(X, X, **self.params)
        # Define QP and solve it with cvxpy
        alpha = cp.Variable(N_tr)
        objective = cp.Maximize(2*alpha.T@y - cp.quad_form(alpha, K))
        constraints = [0 <= cp.multiply(y,alpha), cp.multiply(y,alpha) <= 1/(2*self.lmbd*N_tr)]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        self.alpha = alpha.value

        return self


class MultipleKernel:
    def __init__(self):
        #TODO


    def predict(self):
        #TODO