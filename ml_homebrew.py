"""Module to hold custom machine learning algorithms and functions."""
import numpy as np
import matplotlib.pyplot as plt


class MyError(Exception):
    """Simple error class."""

    def __init__(self, value):
        """Initiate error with value."""
        self.value = value

    def __str__(self):
        """Define representation of error."""
        return repr(self.value)


class PerceptronParameters():
    """Class to hold parameters used in perceptron algorithm."""

    def __init__(self, to_plot=0, to_debug=0, to_pla=0):
        """Get parameters."""
        self.to_plot = to_plot
        self.to_debug = to_debug
        self.to_pla = to_pla

    def __repr__(self):
        """Define representation."""
        le_str = ("PerceptronParameters instance with properties:" +
                  "\nto_plot: " + str(self.to_plot) +
                  "\nto_debug: " + str(self.to_debug) +
                  "\nto_pla: " + str(self.to_pla))
        return le_str


def perceptron_train(X, y, N=100, r=1, v0=[], params=PerceptronParameters()):
    """Train a perceptron learning algorithm with some input data.

    Arguments
    ---------
    X:              Input training data, as an N x d or N x d+1 numpy array
    y:              Input training labels, as an N x 1 numpy array
    [N]:            Maximum number of iterations to loop over
    [r]:            Learning rate: scalar value between 0 and 1
    [v0]:           Initial guess at classification boundary,
                    zeros if unspecified.
    [params:]       Parameter object with properties:
        to_plot:        Boolean for whether to plot,
                        default 0
        to_debug:       Boolean for whether to display debug info,
                        default 0
        to_pla:         Boolean for whether to convert perceptron to pocket
                        learning algorithm, default 0

    Outputs
    -------
    v:          Normal to perceptron classification boundary,
                a 1 x d+1 numpy array
    E_t:        Training data error, scalar in range 0 to 1
    """

    # check learning rate is between 0 and 1
    if r > 1 or r <= 0:
        raise MyError("Require: 0 < r <= 1")

    # get number of data points, n, and dimensionality, d
    n, d = np.shape(X)

    # augment data, if required
    if all((X[:, -1]) == 1):
        d = d - 1
    else:
        X = np.hstack([X, np.ones([n, 1])])

    # initialise classification boundary, if required
    if len(v0) == 0:
        v0 = np.zeros([d + 1])
    v = v0

    # initialise error vector, E
    E = 1

    # to keep track of best optimisation boundary and corresponding index in E
    v_opt = v
    E_opt = 1

    for i in range(0, N):
        # get classification labels, h_x
        h = nrz_classifier(v)
        h_x = h(X)
        # only update where h_x != y
        v = v + r * np.dot(y * (h_x != y), X)
        E = sum(y != h_x) / float(n)
        # update optimum boundary
        if E < E_opt:
            E_opt = E
            v_opt = v

    if params.to_pla:
        return v_opt, E_opt
    else:
        return v, E

    # plot output if specified to do so
    if params.to_plot:
        z = 0.5 * (y + 1)
        cols = plt.cm.coolwarm(z)
        plt.scatter(X[:, 0], X[:, 1], color=cols)
        plt.show()


def perceptron_test():
    """Test a perceptron learning algorithm."""
    # num data points
    n = 40
    # dimensionality
    d = 2
    # dummy data
    X = 2 * np.random.rand(n, d) - 1
    # get boundary and classifier
    dummy_boundary = np.array([-1, 1, 0])
    print "true v: ", dummy_boundary
    f = nrz_classifier(dummy_boundary)
    # augment data
    X2 = np.hstack([X, np.ones([n, 1])])
    # get labels
    y = f(X2)

    # instanciate parameter object:
    params = PerceptronParameters(to_plot=1, to_debug=0, to_pla=1)

    v_out, _ = perceptron_train(X, y, N=50, r=1, params=params)

    print "returned v: ", v_out


def nrz_classifier(coef):
    """Return {-1, 1} classifier from boundary normal (coef)."""
    def inner(dat):
        in_prod = np.inner(dat, coef)
        return (in_prod > 0) * 2 - 1
    return inner


if __name__ == "__main__":
    choice = 0
    if choice == 0:
        perceptron_test()
    else:
        raise MyError("Invalid choice")
