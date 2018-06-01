"""
Kernel density estimation with Mixture of Gaussians.
"""

import numpy as np
import math
from sklearn.neighbors import KernelDensity

from decimal import Decimal


def model(data_a, data_b, sigma):
    """
    This function computes a kernel density estimator from a set of
    example training data, by placing Gaussian kernels on each training
    data.

    Args:
        data_a: Train split with the shape (N x d)
        data_b: Val/test split with the shape (N x d)
        sigma: Bandwidth

    Returns:
        L: Mean
    """

    len_a = data_a.shape[0]
    len_b = data_b.shape[0]
    # d is the dimension of the data
    d = len(data_a[0])
    # mu_ij = x_ij from the training set
    mu = data_a.astype(np.float64)

    log_prob_sum = 0
    term_2 = -(0.5*d)*(math.log(2*math.pi*(sigma**2)))
    cons_log_k = (math.log(len_a))
    for i in range(len_a):
        term_1 = np.sum((-(data_b[i]-mu)**2)/(2*(sigma**2)), axis=1)
        term_1_exp = [(Decimal(t).exp()) for t in term_1]
        inner_sum = sum(term_1_exp)

        log_prob = term_2 - cons_log_k + float(inner_sum.ln())
        log_prob_sum += log_prob
    L = log_prob_sum/len_b

    return L


def sklearn_kde(data_a, data_b, sigma):
    """Used for comparison purposes.
    """

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=sigma, kernel='gaussian')
    kde.fit(data_a)
    # score_samples returns the log of the probability density
    logprob = kde.score_samples(data_b)

    # print('Log probability -> ', logprob)
    sum_x = np.sum(logprob)/data_b.shape[0]

    return sum_x
