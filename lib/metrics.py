# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '01/2013'

"""Usefull metrics.
"""


import numpy as np
import scipy.sparse as sp


EPSILON = 1.e-8


def generalized_KL(x, y, eps=EPSILON, axis=None):
    return (np.multiply(x, np.log(np.divide(x + eps, y + eps))) - x + y
            ).sum(axis=axis)


def hoyer_sparseness(X, axis=-1, eps=EPSILON):
    """Computes Hoyer's measure of sparsity on the given axis.
    The result is then averaged over other axes.
    """
    sqrt_n = np.sqrt(X.shape[axis])
    norm_1 = np.abs(X).sum(axis=axis) + eps
    norm_2 = np.sqrt(np.square(X).sum(axis=axis)) + eps
    return (sqrt_n - np.average(norm_1 / norm_2)) / (sqrt_n - 1)


def entropy(vect):
    return -np.sum(vect * np.log(vect + 1 * (vect == 0)))


def mutual_information(histo):
    assert(histo.ndim == 2)
    assert(np.allclose(histo.sum(), 1.))
    zeros = (histo == 0) * 1  # To avoid nans
    return (histo * np.log((histo + zeros) / (
        histo.sum(axis=0)[np.newaxis, :] * histo.sum(axis=1)[:, np.newaxis]
        + zeros))).sum()


def conditional_entropy(histo, axis=0):
    assert(np.allclose(histo.sum(), 1.))
    # H(Y | X) = H(X, Y) - H(X)
    # given axis gives the variable over which to condition, thus
    # to compute H(X) one must marginalize over all others.
    # Sums on all but one dimension (compute the marginalized
    # over all other dims)
    s = histo.swapaxes(axis, -1)
    s = s.reshape((np.prod(s.shape[:-1]), s.shape[-1])).sum(axis=0)
    return entropy(histo) - entropy(s)


def kl_div(a, b, axis=-1, eps=EPSILON, normalize=False):
    if normalize:
        a /= np.expand_dims(a.sum(axis=axis), axis)
        b /= np.expand_dims(b.sum(axis=axis), axis)
    return generalized_KL(a, b, eps=EPSILON, axis=axis)


def rev_kl_div(a, b, **kwargs):
    return kl_div(b, a, **kwargs)


def sym_kl_div(*args, **kwargs):
    return .5 * (kl_div(*args, **kwargs) + rev_kl_div(*args, **kwargs))


def frobenius(a, b, axis=-1):
    return np.sqrt(np.square(a - b).sum(axis=axis))


def cosine_similarity(a, b, axis=-1):
    ab = np.multiply(a, b).sum(axis=axis)
    return ab / (
            np.sqrt(np.square(a).sum(axis=axis) * np.square(b).sum(axis=axis))
            + (ab == 0)
            )  # returns 0 when a == 0 or b == 0


def cosine_diff(a, b, axis=-1):
    return -cosine_similarity(a, b, axis=axis)


def real_sparsity(X):
    """Computes real sparsity of the given matrix
    (ratio of zero coefficients).
    """
    if sp.issparse(X):
        non_zero = len(X.nonzero()[0])
    else:
        non_zero = (X != 0).sum()
    return  1 - float(non_zero) / np.prod(X.shape)
