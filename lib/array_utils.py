import numpy as np
import scipy.sparse as sp


def safe_hstack(blocks):
    if any([sp.issparse(b) for b in blocks]):
        return sp.hstack(blocks)
    else:
        return np.hstack(blocks)


def safe_vstack(Xs):
    if any(sp.issparse(X) for X in Xs):
        return sp.vstack(Xs)
    else:
        return np.vstack(Xs)


def normalize_sum(a, axis=0, eps=1.e-16):
    if axis >= len(a.shape):
        raise ValueError
    return a / (eps + np.expand_dims(np.sum(a, axis=axis), axis))


def normalize_features(X):
    """Normalizes columns and remove columns with 0 sum.
    Returns sparse matrices as csc.
    """
    sums = np.asarray(X.sum(axis=0)).flatten()
    if sp.issparse(X):
        X.eliminate_zeros()
        x = X.copy().tocsr()
        x.data /= sums[x.indices]
        y = x.tocsc()  # For more efficient column slicing
    else:
        y = X.copy()
        # Adds 1 when s == 0 to avoid runtime warning
        y /= (1. * (sums == 0) + sums)[np.newaxis, :]
    # Remove columns of 0s
    return y[:, sums.nonzero()[0]]
