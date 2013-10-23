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


class GrowingLILMatrix(sp.lil_matrix):

    def __init__(self):
        sp.lil_matrix.__init__(self, (1, 1))
        # By default format = self.__class__.__name__[:3] which is unknown
        # from sparse matrix printer.
        self.format = 'lil'
        self._shape = (0, 0)

    def add_row(self, row):
        """Adds a row to the matrix from a list or array."""
        sparse_row = sp.lil_matrix(row)
        if self._shape == (0, 0):  # Empty GrowingLILMatrix
            # Init matrix with row
            self._shape = sparse_row._shape
            self.rows = sparse_row.rows
            self.data = sparse_row.data
        else:
            # Update shape
            nb_col = max(len(row), self.shape[1])
            self._shape = (1 + self._shape[0], nb_col)
            # Add the row
            self.rows = np.concatenate([self.rows, sparse_row.rows[[0]]])
            self.data = np.concatenate([self.data, sparse_row.data[[0]]])
