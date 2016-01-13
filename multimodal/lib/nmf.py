""" Non-negative matrix factorization for I divergence

    This code was originally written as an alternative to the implementation
    of NMF for Frobenius error minimization in the scikit-learn project.
    This file is a standalone version of that code.
    See http://github.com/omangin/scikit-learn/tree/klnmf
    for an implementation integrated in scikit-learn.
"""

# Author: Olivier Mangin <olivier.mangin@inria.fr>


import sys

import numpy as np
import scipy.sparse as sp

from .metrics import generalized_KL
from .array_utils import normalize_sum
from .sklearn_utils import atleast2d_or_csr, safe_sparse_dot


def check_non_negative(X, whom):
    X = X.data if sp.issparse(X) else X
    if (X < 0).any():
        raise ValueError("Negative values in data passed to %s" % whom)


def _scale(matrix, factors, axis=0):
    """Scales line or columns of a matrix.

    Parameters
    ----------
    :param matrix: 2-dimensional array
    :param factors: 1-dimensional array
    :param axis: 0: columns are scaled, 1: lines are scaled
    """
    if not (len(matrix.shape) == 2):
        raise ValueError(
                "Wrong array shape: %s, should have only 2 dimensions."
                % str(matrix.shape))
    if axis not in (0, 1):
        raise ValueError('Wrong axis, should be 0 (scaling lines)\
                or 1 (scaling columns).')
    # Transform factors given as columne shaped matrices
    factors = np.squeeze(np.asarray(factors))
    if axis == 1:
        factors = factors[:, np.newaxis]
    return np.multiply(matrix, factors)


def _special_sparse_dot(a, b, refmat):
    """Computes dot product of a and b on indices where refmat is nonnzero
    and returns sparse csr matrix with same structure than refmat.

    First calls to eliminate_zeros on refmat which might modify the structure
    of refmat.

    Params
    ------
    a, b: dense arrays
    refmat: sparse matrix

    Dot product of a and b must have refmat's shape.
    """
    refmat.eliminate_zeros()
    ii, jj = refmat.nonzero()
    dot_vals = np.multiply(a[ii, :], b.T[jj, :]).sum(axis=1)
    c = sp.coo_matrix((dot_vals, (ii, jj)), shape=refmat.shape)
    return c.tocsr()


class KLdivNMF(object):
    """Non negative factorization with Kullback Leibler divergence cost.

    Parameters
    ----------
    n_components: int or None
        Number of components, if n_components is not set all components
        are kept

    init:  'nndsvd' |  'nndsvda' | 'nndsvdar' | 'random'
        Method used to initialize the procedure.
        Default: 'nndsvdar' if n_components < n_features, otherwise random.
        Valid options::

            'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
                initialization (better for sparseness)
            'nndsvda': NNDSVD with zeros filled with the average of X
                (better when sparsity is not desired)
            'nndsvdar': NNDSVD with zeros filled with small random values
                (generally faster, less accurate alternative to NNDSVDa
                for when sparsity is not desired)
            'random': non-negative random matrices

    tol: double, default: 1e-4
        Tolerance value used in stopping conditions.

    max_iter: int, default: 200
        Number of iterations to compute.

    subit: int, default: 10
        Number of sub-iterations to perform on W (resp. H) before switching
        to H (resp. W) update.

    Attributes
    ----------
    `components_` : array, [n_components, n_features]
        Non-negative components of the data

    random_state : int or RandomState
        Random number generator seed control.

    Examples
    --------

    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from multimodal.lib.nmf import KLdivNMF
    >>> model = KLdivNMF(n_components=2, init='random', random_state=0)
    >>> model.fit(X) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    KLdivNMF(eps=1e-08, init='random', max_iter=200, n_components=2,
            random_state=0, subit=10, tol=1e-06)
    >>> model.components_
    array([[ 0.50303234,  0.49696766],
           [ 0.93326505,  0.06673495]])

    Notes
    -----
    This implements

    Lee D. D., Seung H. S., Learning the parts of objects by non-negative
      matrix factorization. Nature, 1999
    """

    def __init__(self, n_components=None, tol=1e-6, max_iter=200, eps=1.e-8,
                 subit=10, random_state=None):
        self.n_components = n_components
        self._init_dictionary = None
        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter
        self.eps = eps
        # Only for gradient updates
        self.subit = subit

    def _init(self, X):
        n_samples, n_features = X.shape
        if self._init_dictionary is None:
            H_init = normalize_sum(np.abs(np.random.random(
                (self.n_components, n_features))) + .01, axis=1)
        else:
            assert(self._init_dictionary.shape ==
                   (self.n_components, n_features))
            H_init = self._init_dictionary
        W_init = X.dot(H_init.T)
        return W_init, H_init

    def fit_transform(self, X, y=None, weights=1., _fit=True,
                      return_errors=False, scale_W=False):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be decomposed

        weights: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Weights on the cost function used as coefficients on each
            element of the data. If smaller dimension is provided, standard
            numpy broadcasting is used.

        return_errors: boolean
            if True, the list of reconstruction errors along iterations is
            returned

        scale_W: boolean (default: False)
            Whether to force scaling of W during updates. This is only relevant
            if components are normalized.

        _fit: if True (default), update the model, else only compute transform

        Returns
        -------
        data: array, [n_samples, n_components]
            Transformed data

        or (data, errors) if return_errors
        """
        X = atleast2d_or_csr(X)
        check_non_negative(X, "NMF.fit")

        n_samples, n_features = X.shape

        if not self.n_components:
            self.n_components = n_features

        W, H = self._init(X)

        if _fit:
            self.components_ = H

        prev_error = np.Inf
        tol = self.tol * n_samples * n_features

        if return_errors:
            errors = []

        for n_iter in range(1, self.max_iter + 1):
            # Stopping condition
            error = self.error(X, W, self.components_, weights=weights)
            if prev_error - error < tol:
                break
            prev_error = error

            if return_errors:
                errors.append(error)

            W = self._update(X, W, _fit=_fit)

        if n_iter == self.max_iter and tol > 0:
            sys.stderr.write("Warning: Iteration limit reached during fit\n")

        if return_errors:
            return W, errors
        else:
            return W

    def _update(self, X, W, _fit=True, scale_W=False, eps=1.e-8):
        """Perform one update iteration.

        Updates components if _fit and returns updated coefficients.

        Params:
        -------
            _fit: boolean (default: True)
                Whether to update components.

            scale_W: boolean (default: False)
                Whether to force scaling of W. This is only relevant if
                components are normalized.
        """
        if scale_W:
            # This is only relevant if components are normalized.
            # Not always usefull but might improve convergence speed:
            # Scale W lines to have same sum than X lines
            W = _scale(normalize_sum(W, axis=1), X.sum(axis=1), axis=1)
        Q = self._Q(X, W, self.components_, eps=eps)
        # update W
        W = self._updated_W(X, W, self.components_, Q=Q)
        if _fit:
            # update H
            self.components_ = self._updated_H(X, W, self.components_, Q=Q)
        return W

    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be decomposed

        Returns
        -------
        self
        """
        self.fit_transform(X, **params)
        return self

    def transform(self, X, **params):
        """Transform the data X according to the fitted NMF model

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be transformed by the model

        Returns
        -------
        data: array, [n_samples, n_components]
            Transformed data
        """
        self._init_dictionary = self.components_
        params['_fit'] = False
        return self.fit_transform(X, **params)

    # Helpers for beta divergence and related updates

    # Errors and performance estimations

    def error(self, X, W, H=None, weights=1., eps=1.e-8):
        X = atleast2d_or_csr(X)
        if H is None:
            H = self.components_
        if sp.issparse(X):
            WH = _special_sparse_dot(W, H, X)
            # Avoid computing all values of WH to get their sum
            WH_sum = np.sum(np.multiply(np.sum(W, axis=0), np.sum(H, axis=1)))
            return (np.multiply(
                X.data,
                np.log(np.divide(X.data + eps, WH.data + eps))
                )).sum() - X.data.sum() + WH_sum
        else:
            return generalized_KL(X, np.dot(W, H))

    # Projections

    def scale(self, W, H, factors):
        """Scale W columns and H rows inversely, according to the given
        coefficients.
        """
        safe_factors = factors + self.eps
        s_W = _scale(W, safe_factors, axis=0)
        s_H = _scale(H, 1. / safe_factors, axis=1)
        return s_W, s_H

    # Update rules

    @classmethod
    def _Q(cls, X, W, H, eps=1.e-8):
        """Computes X / (WH)
           where '/' is element-wise and WH is a matrix product.
        """
        # X should be at least 2D or csr
        if sp.issparse(X):
            WH = _special_sparse_dot(W, H, X)
            WH.data = (X.data + eps) / (WH.data + eps)
            return WH
        else:
            return np.divide(X + eps, np.dot(W, H) + eps)

    @classmethod
    def _updated_W(cls, X, W, H, weights=1., Q=None, eps=1.e-8):
        if Q is None:
            Q = cls._Q(X, W, H, eps=eps)
        W = np.multiply(W, safe_sparse_dot(Q, H.T))
        return W

    @classmethod
    def _updated_H(cls, X, W, H, weights=1., Q=None, eps=1.e-8):
        if Q is None:
            Q = cls._Q(X, W, H, eps=eps)
        H = np.multiply(H, safe_sparse_dot(W.T, Q))
        H = normalize_sum(H, axis=1)
        return H
