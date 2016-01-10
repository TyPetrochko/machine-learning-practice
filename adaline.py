import numpy as np

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
    """A python adaptive linear neuron classifier!
        
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

