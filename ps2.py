"""
Module to hold homework set 2 problems.

Problems requiring computational solutions are 1, 2.
"""

import matplotlib.pyplot as plt
import numpy as np


def heof(n_coins=1000, n_flips=10, n_experiments=100000):
    """Solve problems 1 and 2 on the Heoffding Inequality."""
    # create array to hold all coins and all flips for one
    # experiment
    sz = (n_coins, n_flips)

    # distributions
    nu_firsts = np.zeros(n_experiments)
    nu_rands = np.zeros(n_experiments)
    nu_mins = np.zeros(n_experiments)

    print 'Start...'

    for i in range(0, n_experiments):

        rands = np.random.rand(*sz)

        coins = np.zeros(sz)
        coins[rands > 0.5] = 1

        nu = np.average(coins, 1)

        nu_first = nu[0]

        rnd_idx = np.round((n_coins - 1) * np.random.rand())
        nu_rand = nu[rnd_idx]

        nu_min = np.ndarray.min(nu)

        # add to distributions
        nu_firsts[i] = np.around(nu_first, 2)
        nu_rands[i] = np.around(nu_rand, 2)
        nu_mins[i] = np.around(nu_min, 2)

    print 'Average of nu_min =', np.average(nu_mins)

    # plot distributions
    plt.hist(nu_firsts, bins=10, histtype='stepfilled')
    plt.hist(nu_rands, bins=10, histtype='stepfilled', alpha=0.67)
    plt.hist(nu_mins, bins=10, histtype='stepfilled', alpha=0.33)
    plt.show()


def lin_reg():
    """Solve problems 5 and 6 on Linear Regression."""
    def classifier(coef):
        def map(dat):
            in_prod = np.inner(dat, coef)
            return (in_prod > 0) * 2 - 1
        return map
    x = np.array([1, 1, 1])
    f = classifier(np.array([1, 1, 1]))
    g = classifier(np.array([-1, -1, -1]))
    print(f(x))
    print(g(x))


def perform_chosen_function(choice=0):
    """Perform main."""
    if choice == 0:
        heof()
    elif choice == 1:
        lin_reg()


if __name__ == "__main__":
    perform_chosen_function(choice=1)
