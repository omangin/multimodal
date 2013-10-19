# -*- coding: utf-8 -*-


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '07/2011'


import numpy as np
from scipy.cluster.vq import vq


EPSILON = 1.e-9


def square_distances(vectors, centroids):
    """Returns the n x k matrix of square_distances between each of the n
    vectors and the k centroids.

    :params vectors; array, (n, dim)
    :param centroids: array, (k, dim)
    """
    assert vectors.shape[1] == centroids.shape[1]
    return np.sum(np.square(
        vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :]),
        axis=-1)


def associate(vectors, centroids):
    """Returns associations of vectors on centroids.
    """
    dists = square_distances(vectors, centroids)
    assoc = np.argmin(dists, axis=1)
    return assoc


def soft_associate(vectors, centroids, alpha=.5):
    """Returns a matrix (n, k) of association wights, defined as:
        w_i,c = e^(- alpha * d_i,c^2 / d_min,i^2) / sum for normalization
    """
    # Get dists and dist to closest centroid
    #dists = np.sqrt(square_distances(vectors, centroids))
    dists = square_distances(vectors, centroids)
    min_dist = np.expand_dims(np.min(dists, axis=1), 1)
    # Compute weights
    weights = np.exp(-alpha * dists / (EPSILON + min_dist))
    # Normalize weights to sum to 1
    sums = np.sum(weights, axis=1)
    return weights / np.expand_dims(sums, 1)


def get_histos(x, cb, soft=None):
    k, d = cb.shape
    n, d = x.shape
    if soft is None:
        c, _ = vq(x, cb)
        h = np.eye(k)
        h = h[c, :]
    else:
        h = soft_associate(x, cb, alpha=soft)
    return h
