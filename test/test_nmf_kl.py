import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
import scipy.sparse as sp

from multimodal.lib import nmf
from multimodal.lib.metrics import generalized_KL


def random_NN_matrix(shape):
    return np.abs(np.random.random(shape))


def random_NN_sparse(h, w, density):
    r = sp.rand(h, w, density)
    r.data = np.abs(r.data)
    return r


def is_NN(a):
    return np.all(a >= 0)


class TestScale(unittest.TestCase):

    def test_shape(self):
        mtx = np.zeros((3, 4))
        fact = np.zeros((3,))
        self.assertEqual(mtx.shape, nmf._scale(mtx, fact, axis=1).shape)

    def test_error_on_wrong_axis(self):
        mtx = np.zeros((3, 4))
        fact = np.zeros((4,))
        with self.assertRaises(ValueError):
            nmf._scale(mtx, fact, axis=3)

    def test_error_on_3D_array(self):
        mtx = np.zeros((3, 4, 6))
        fact = np.zeros((3,))
        with self.assertRaises(ValueError):
            nmf._scale(mtx, fact, axis=1)

    def test_error_on_1D_array(self):
        mtx = np.zeros((3,))
        fact = np.zeros((3,))
        with self.assertRaises(ValueError):
            nmf._scale(mtx, fact, axis=1)

    def test_error_on_wrong_factor_shape(self):
        mtx = np.zeros((3, 4))
        fact = np.zeros((2,))
        with self.assertRaises(ValueError):
            nmf._scale(mtx, fact, axis=1)

    def test_scale_lines(self):
        mtx = np.array([[1, 2, 3], [4, 5, 6]])
        fact = np.array([2, 3])
        scaled = nmf._scale(mtx, fact, axis=1)
        ok = np.array([[2, 4, 6], [12, 15, 18]])
        assert_array_almost_equal(ok, scaled)

    def test_scale_columns(self):
        mtx = np.array([[1, 2, 3], [4, 5, 6]])
        fact = np.array([3, 2, 1])
        scaled = nmf._scale(mtx, fact, axis=0)
        ok = np.array([[3, 4, 3], [12, 10, 6]])
        assert_array_almost_equal(ok, scaled)


class TestError(unittest.TestCase):

    n_samples = 20
    n_components = 3
    n_features = 30

    def setUp(self):
        self.X = random_NN_sparse(self.n_samples, self.n_features, .1)
        self.W = random_NN_matrix((self.n_samples, self.n_components))
        self.H = random_NN_matrix((self.n_components, self.n_features))
        self.nmf = nmf.KLdivNMF(n_components=3, tol=1e-4,
            max_iter=200, eps=1.e-8, subit=10)
        self.nmf.components_ = random_NN_matrix((self.n_components,
                self.n_features))

    def test_error_is_gen_kl(self):
        Xdense = self.X.todense()
        err = self.nmf.error(Xdense, self.W, H=self.H)
        kl = generalized_KL(Xdense, self.W.dot(self.H))
        assert_array_almost_equal(err, kl)

    def test_error_sparse(self):
        err_dense = self.nmf.error(self.X.todense(), self.W, H=self.H)
        err_sparse = self.nmf.error(self.X, self.W, H=self.H)
        assert_array_almost_equal(err_dense, err_sparse)

    def test_error_is_gen_kl_with_compenents(self):
        Xdense = self.X.todense()
        err = self.nmf.error(Xdense, self.W)
        kl = generalized_KL(Xdense, self.W.dot(self.nmf.components_))
        assert_array_almost_equal(err, kl)


class TestUpdates(unittest.TestCase):

    n_samples = 20
    n_components = 3
    n_features = 30

    def setUp(self):
        self.X = random_NN_matrix((self.n_samples, self.n_features))
        self.W = random_NN_matrix((self.n_samples, self.n_components))
        self.H = random_NN_matrix((self.n_components, self.n_features))
        self.nmf = nmf.KLdivNMF(n_components=3, tol=1e-4,
            max_iter=200, eps=1.e-8, subit=10)
        self.nmf.components_ = self.H

    def test_W_remains_NN(self):
        W = self.nmf._updated_W(self.X, self.W, self.H)
        self.assertTrue(is_NN(W))

    def test_H_remains_NN(self):
        H = self.nmf._updated_H(self.X, self.W, self.H)
        self.assertTrue(is_NN(H))

    def test_decreases_KL(self):
        dkl_prev = self.nmf.error(self.X, self.W)
        W = self.nmf._update(self.X, self.W, _fit=True)
        dkl_next = self.nmf.error(self.X, W)
        self.assertTrue(dkl_prev > dkl_next)

    def test_no_compenents_update(self):
        self.nmf._update(self.X, self.W, _fit=False)
        self.assertTrue((self.nmf.components_ == self.H).all())


class TestSparseUpdates(TestUpdates):
    """Checks that updates are OK with sparse input.
    """

    def setUp(self):
        self.X = random_NN_sparse(self.n_samples, self.n_features, .5).tocsr()
        self.W = random_NN_matrix((self.n_samples, self.n_components))
        self.H = random_NN_matrix((self.n_components, self.n_features))
        self.nmf = nmf.KLdivNMF(n_components=3, tol=1e-4,
            max_iter=200, eps=1.e-8, subit=10)
        self.nmf.components_ = self.H


class TestFitTransform(unittest.TestCase):

    def setUp(self):
        self.nmf = nmf.KLdivNMF(n_components=3, tol=1e-6,
            max_iter=200, eps=1.e-8, subit=10)

    def test_cv(self):
        X = random_NN_matrix((10, 5))
        W, errors = self.nmf.fit_transform(X, return_errors=True)
        # Last errors should be very close
        self.assertTrue(abs(errors[-1] - errors[-2]) < errors[0] * 1.e-2)

    def test_zero_error_on_fact_data(self):
        X = np.dot(random_NN_matrix((5, 2)), random_NN_matrix((2, 3)))
        W, errors = self.nmf.fit_transform(X, return_errors=True)
        self.assertTrue(errors[-1] < errors[0] * 1.e-3)

    def test_no_compenents_update(self):
        components = random_NN_matrix((3, 5))
        self.nmf.components_ = components
        self.nmf.fit_transform(random_NN_matrix((10, 5)), components,
            _fit=False)
        self.assertTrue((self.nmf.components_ == components).all())


class TestSparseDot(unittest.TestCase):

    def setUp(self):
        self.ref = sp.rand(5, 6, .3).tocsr()
        self.a = np.random.random((5, 7))
        self.b = np.random.random((7, 6))

    def test_indices(self):
        """Test that returned sparse matrix has same structure than refmat.
        """
        ab = nmf._special_sparse_dot(self.a, self.b, self.ref)
        self.assertTrue((ab.indptr == self.ref.indptr).all()
                and (ab.indices == self.ref.indices).all())

    def test_correct(self):
        ok = np.multiply(np.dot(self.a, self.b), (self.ref.todense() != 0))
        ans = nmf._special_sparse_dot(self.a, self.b, self.ref).todense()
        assert_array_almost_equal(ans, ok)
