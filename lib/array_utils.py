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
    """
    sums = np.asarray(X.sum(axis=0)).flatten()
    if sp.issparse(X):
        X.eliminate_zeros()
        x = X.copy().tocsr()
        x.data /= sums[x.indices]
        x = x.tocsc()
        m, n = x.shape
        y = sp.csc_matrix((m, (sums != 0).sum()))
        y.data = x.data
        y.indices = x.indices
        y.indptr = np.asarray(np.hstack([x.indptr[sums.nonzero()[0]],
                                         [len(y.data)]
                                         ]), dtype=y.indptr.dtype)
    else:
        y = X.copy()
        # Adds 1 when s == 0 to avoid runtime warning
        y /= (1. * (sums == 0) + sums)[np.newaxis, :]
        y = y[:, sums.nonzero()[0]]
    return y
