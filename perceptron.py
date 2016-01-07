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

class Perceptron(object):
    """A python perceptron!
        
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

        for _ in range(self.n_iter):
            'Fit training data'
            errors = 0
            
            # Iterate through all training data
            #   xi is input array, target is desired output
            for xi, target in zip(X, y):
                # Calculate (training rate * miss) a.k.a update amount
                update = self.eta * (target - self.predict(xi))

                # To each attribute weight, add the update amount times attribute value
                #
                # I.e. if we predicted -0.8 when we should have gotten 1.0, we will
                # have a large update amount. Thus each weight must be "scaled up"
                # by an amount proportional to update amount and the input attribute.
                #
                # E.g. weights are [3,0 2.4, -0.5], xi is [46.5, 42.3], target is
                # 1, and learning rate is 0.1, then net input is 93.46, target - prediction 
                # is -92.45,thus update amount is -9.245. New weights become: 
                # [-6.245, -19.788, 4.1225] and thus net input NEXT round will become
                # -55.7975, target - prediction is 56.79.
                #
                # (I think the learning rate is a bit too high in this example).
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        'Calculate net input'
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        'Return class label after unit step'
        return np.where(self.net_input(X) >= 0.0, 1, -1)
