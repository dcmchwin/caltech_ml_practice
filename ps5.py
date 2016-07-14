"""Python file for problem set 5 of caltech machine learning course."""

import numpy as np
import argparse

import time

class GradientDescender(object):
    """For problems 4 through 7. Error surface = (ue^v - 2ve^-u)^2."""

    def __init__(self, u=1, v=1, eta=0.1, E=None, gradE=None, E_lim=10**-14):
        """Initiate solver."""
        self.u = u
        self.v = v
        self.eta = eta
        self.E = E
        if gradE is None:
            self.gradE = [None] * 2
        else:
            self.gradE = gradE
        self.n = 0  # counter
        self.E_lim = E_lim

    def get_E(self):
        """Calculate error using given error surface."""
        self.E = (self.u * np.exp(self.v) - 2 * self.v * np.exp(-self.u)) ** 2

    def get_gradE(self):
        """Calculate gradient of E wrt (u, v)."""
        pt1 = self.u * np.exp(self.v) - 2 * self.v * np.exp(-self.u)
        pt2 = np.exp(self.v) + 2 * self.v * np.exp(-self.u)
        pt3 = self.u * np.exp(self.v) - 2 * np.exp(-self.u)

        self.gradE[0] = 2 * pt1 * pt2
        self.gradE[1] = 2 * pt1 * pt3

    def gradient_descent(self):
        """Perform the gradient descent algorithm."""
        self.get_E()
        while self.E >= self.E_lim:
            self.get_gradE()
            self.u -= self.eta * self.gradE[0]
            self.v -= self.eta * self.gradE[1]
            self.get_E()
            self.n += 1
            print 'E:', self.E, '\tgradE:', self.gradE
            print '(u, v):', [self.u, self.v], '\tn:', self.n, '\n'
            time.sleep(0.5)

        return dict(E=self.E, n_steps=self.n, w=[self.u, self.v])

    def coordinate_descent(self):
        """Perform the coordinate descent algorithm."""
        self.get_E()
        while self.n < 15:
            self.get_gradE()
            self.u -= self.eta * self.gradE[0]
            self.get_gradE()
            self.v -= self.eta * self.gradE[1]
            self.get_E()
            self.n += 1

        return dict(E=self.E, n_steps=self.n, w=[self.u, self.v])

def problem5():
    descender = GradientDescender(u=1, v=1)
    results = descender.gradient_descent()
    print results

def problem7():
    descender = GradientDescender(u=1, v=1)
    results = descender.coordinate_descent()
    print results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', default='5',
                        help='problem number to solve')
    args = parser.parse_args()

    problem_dict = {'5': problem5, '7': problem7}

    problem_dict[args.problem]()
