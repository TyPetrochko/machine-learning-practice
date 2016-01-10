import numpy as np
from numpy.random import seed

# Just a description of some of the parameters
#   -- Paramaters --
#
#   X = the (training) data itself (no headers),
#       where each row maps to a single item,
#       and each column is an attribute, like
#       petal length, sepal length, height,
#       etc.
#
#   y = classification data, i.e. each row is the
#       "classification" of the item in that row.
#       For two types of items, like Setosa and
#       Versicolor, you only have two species so
#       y can either be 1 (versicolor) or -1
#       (setosa)
#
#   ... our eventual goal is to map a function that
#   solves/approximates the equation
#
#           A * X = y
#

class AdalineGD(object):
    """A python adaptive linear neuron classifier, using gradient descent!
        
        eta : float, training rate 0.0-1.0
        n_iter : int, num of training passes
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        # initialize all weights to zero (one plus number of attributes)
        self.w_ = np.zeros(1 + X.shape[1])
        # track errors at each iteration
        self.errors_ = []
        # track the costs
        self.cost_ = []

        for i in range(self.n_iter):
            'Fit training data'
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        'Calculate net input'
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        'Return class label after unit step'
        return np.where(self.net_input(X) >= 0.0, 1, -1)

class AdalineSGD(object):
    """A python adaptive linear neuron classifier, using stochastic gradient descent!
        
        eta : float, training rate 0.0-1.0
        n_iter : int, num of training passes
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_init = False # are weights initialized yet?
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """ Fit training data without re-init'ing weights """

        if not self.w_init:
            self._initialize_weights(X.shape[1])
            if y.ravel().shape[0] > 1:
                for xi, target in zip(X, y):
                    self._update_weights(xi, target)
            else:
                self._update_weights(X, y)
            return self

    def _shuffle(self, X, y):
        """ Shuffle training data """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """ Init all weights to zeros """
        self.w_ = np.zeros(1 + m)
        self.w_init = True

    def _update_weights(self, xi, target):
        """ Apply adaline learning rule to update weights """
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def activation(self, X):
        """ Compute linear activation"""
        return self.net_input(X)

    def net_input(self, X):
        'Calculate net input'
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        'Return class label after unit step'
        return np.where(self.net_input(X) >= 0.0, 1, -1)

