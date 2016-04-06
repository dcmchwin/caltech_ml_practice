"""Module to hold mathematical vector utilities."""
import numpy as np


class MyError(Exception):
    """Define custom exceptions."""

    def __init__(self, value):
        """Define init method."""
        self.value = value

    def __str__(self):
        """Define string representation."""
        return repr(self.value)


def n_d_cross(*args):
    """
    Return d-dimensional cross product of vectors.

    input: d-1 numpy vectors of dimension d
    """
    d = len(args[0])
    # print d
    n = len(args)
    # print n

    # error check
    if n != d - 1:
        raise MyError("Number of input vectors must be one less than" +
                      " vector dimension")

    # initialise cross product output
    v = np.zeros(d)

    # define tuple of matrices
    for i in range(0, d):
        A = np.vstack([np.append(a[:i], a[i + 1:]) for a in args])
        v[i] = np.linalg.det(A) * (2 * np.mod(i, 2) - 1)

    return v


def get_pseudo_inverse(x):
    """Return the pseudo-inverse of a n by d numpy array."""
    xt = np.transpose(x)
    xtx = np.dot(xt, x)
    xtx_inv = np.linalg.inv(xtx)
    x_pseudo_inv = np.dot(xtx_inv, xt)
    return x_pseudo_inv


if __name__ == "__main__":
    test = np.array([1, 0, 0])
    test2 = np.array([0, 0, 1])

    l = [test, test2]

    vec = n_d_cross(*l)

    print test, " x ", test2, " = ", vec
