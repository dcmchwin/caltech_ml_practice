"""
Module to hold homework set 2 problems.

Problems requiring computational solutions are 1, 2.
"""

import matplotlib.pyplot as plt
import numpy as np
import vector


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


def lin_reg(n_points=100, n_experiments=10):
    """Solve problems 5 and 6 on Linear Regression."""
    def classifier(coef):
        def inner(dat):
            in_prod = np.inner(dat, coef)
            return (in_prod > 0) * 2 - 1
        return inner

    def get_rand_coords(dim=2, n_points=2):
        return [2 * np.random.rand(dim) - 1 for i in range(0, n_points)]

    def gen_decision_normal():
        f_coords = get_rand_coords()
        f_coords_plus = [np.append(t, [1]) for t in f_coords]
        f_n = vector.n_d_cross(*f_coords_plus)
        f_n = f_n[:-1]
        f_n = f_n / np.linalg.norm(f_n)
        b = np.inner(f_n, f_coords[0])
        f_n = np.append(f_n, b)
        return f_n

    def get_row(np_arr_list, i):
        # take in a list of np arrays and return a certain row thereof
        return [x[i] for x in np_arr_list]

    def scatter_points(x_list, y_list):
        z = [0.5 * (el + 1) for el in y_list]
        cols = plt.cm.coolwarm(z)
        plt.scatter(get_row(x_list, 0), get_row(x_list, 1), color=cols)
        plt.show()

    def get_linreg_boundary(X, y):
        X_pseudo_inv = vector.get_pseudo_inverse(X)
        return np.dot(X_pseudo_inv, y)

    def problem5(n_points, n_experiments, f):
        print "Problem 5..."
        g_n_list = []
        avg_err = 0
        for i in range(0, n_experiments):
            x_list = get_rand_coords(n_points=n_points)
            x_list_2 = [np.append(x, [1]) for x in x_list]
            y_list = [f(x) for x in x_list_2]

            X = np.vstack(x_list_2)
            y = np.hstack(y_list)

            g_n = get_linreg_boundary(X, y)
            g = classifier(g_n)
            g_n_list.append(g_n)

            y_g = [g(x) for x in x_list_2]

            err = sum((y != y_g)) / float(n_points)

            avg_err = avg_err + err / float(n_experiments)

        print "avg_err: ", avg_err
        return g_n_list

    def problem6(n_points, n_experiments, f, g_n_list):
        print "Problem 6..."
        avg_err = 0
        for i in range(0, n_experiments):
            x_list = get_rand_coords(n_points=n_points)
            x_list_2 = [np.append(x, [1]) for x in x_list]
            g_n = g_n_list[i]
            g = classifier(g_n)

            y_f = np.hstack([f(x) for x in x_list_2])
            y_g = [g(x) for x in x_list_2]

            err = sum((y_f != y_g)) / float(n_points)

            avg_err = avg_err + err / float(n_experiments)
        print "avg_err: ", avg_err

    f_n = gen_decision_normal()
    f = classifier(np.array(f_n))

    g_n_list = problem5(n_points, n_experiments, f)
    problem6(1000, 1000, f, g_n_list)



def perform_chosen_function(choice=0):
    """Perform main."""
    if choice == 0:
        heof()
    elif choice == 1:
        lin_reg(n_experiments=1000)


if __name__ == "__main__":
    perform_chosen_function(choice=1)
