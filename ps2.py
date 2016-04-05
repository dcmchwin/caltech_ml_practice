import matplotlib.pyplot as plt
import numpy as np


class Homework2(object):
    """
    Class to hold homework set 2 problems
    Problems requiring computational solutions are 1, 2, 
    """
    @staticmethod
    def heof(n_coins=1000, n_flips=10, n_experiments=100000):
        """
        covers problems 1 and 2 on the Heoffding Inequality
        """

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

    @staticmethod
    def err_and_noise():
        """
        covers problems 3 and 4 on error and err_and_noise
        """


Homework2.heof()
