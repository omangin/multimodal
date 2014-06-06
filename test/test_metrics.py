# encoding: utf-8


import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from multimodal.lib.metrics import (generalized_KL, hoyer_sparseness, entropy,
                                    mutual_information, conditional_entropy,
                                    cosine_similarity)


def random_NN_matrix(h, w):
    return np.abs(np.random.random((h, w)))


class TestGenenralizedKL(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.x = random_NN_matrix(10, 15)
        self.y = random_NN_matrix(10, 15)

    def test_returns_scalar(self):
        self.assertTrue(np.isscalar(generalized_KL(self.x, self.y)))

    def test_raises_ValueError_0(self):
        with self.assertRaises(ValueError):
            generalized_KL(self.x[:-1, :], self.y)

    def test_raises_ValueError_1(self):
        with self.assertRaises(ValueError):
            generalized_KL(self.x[:, 1:], self.y)

    def test_is_NN(self):
        self.assertTrue(generalized_KL(self.x, self.y) >= 0)

    def test_is_0_on_same(self):
        assert_array_almost_equal(generalized_KL(self.x, self.x), 0)

    def test_is_1_homogenous(self):
        dkl = generalized_KL(self.x, self.y)
        a = np.random.random()
        adkl = generalized_KL(a * self.x, a * self.y)
        assert_array_almost_equal(a * dkl, adkl, decimal=5)

    def test_values(self):
        x = np.zeros((4, 2))
        x[1, 1] = 1
        y = .5 * np.ones((4, 2))
        dkl = generalized_KL(x, y)
        ok = np.log(2.) + 3.
        assert_array_almost_equal(dkl, ok)

    def test_axis(self):
        dkl = generalized_KL(self.x, self.y, axis=1)
        self.assertEqual(dkl.shape, (10,))
        dkl = generalized_KL(self.x, self.y, axis=0)
        self.assertEqual(dkl.shape, (15,))


class TestHoyerSparseness(unittest.TestCase):

    def test_is_in_0_1(self):
        x = np.random.random((5, 7))
        s = hoyer_sparseness(x)
        assert(np.all(0 <= s <= 1))

    def test_axis(self):
        x = np.array([[1, 1],
                      [0, 0]])
        s1 = hoyer_sparseness(x, axis=0)
        self.assertEqual(s1, 1.)

    def test_is_0_on_flat(self):
        x = np.ones((1, 4))
        self.assertAlmostEqual(hoyer_sparseness(x), 0.)

    def test_is_1_on_spike(self):
        x = np.zeros((1, 4))
        x[0, 2] = 10.
        self.assertEqual(hoyer_sparseness(x), 1.)

    def test_zero(self):
        x = np.zeros((2, 3))
        s1 = hoyer_sparseness(x)
        self.assertEqual(s1, 1.)


class TestEntropy(unittest.TestCase):

    def test_is_positive(self):
        p = np.random.dirichlet([3.] * 10)
        self.assertTrue(entropy(p) >= 0)

    def test_is_symetric(self):
        p = np.random.dirichlet([3.] * 10)
        e1 = entropy(p)
        np.random.shuffle(p)
        e2 = entropy(p)
        self.assertAlmostEqual(e1, e2)

    def test_is_log_n_on_equi(self):
        p = np.ones((10,)) / 10.
        self.assertAlmostEqual(entropy(p), np.log(10.))

    def test_is_0_on_deterministic(self):
        self.assertEqual(entropy(np.array([0, 1])), 0)


class TestMutualInformation(unittest.TestCase):

    def test_is_positive(self):
        p = np.random.dirichlet([3.] * 10).reshape((2, 5))
        self.assertTrue(mutual_information(p) >= 0)

    def test_is_0_on_indep(self):
        p = np.random.dirichlet([3.] * 10)
        p = .5 * np.vstack([p, p])
        self.assertAlmostEqual(mutual_information(p), 0)

    def test_fits_relation_to_entropy(self):
        p = np.random.dirichlet([3.] * 10).reshape((2, 5))
        info1 = mutual_information(p)
        info2 = entropy(p.sum(axis=0)) + entropy(p.sum(axis=1)) - entropy(p)
        self.assertAlmostEqual(info1, info2)

    def test_is_one_bit(self):
        p = np.array([[.5, 0.], [0., .5]])
        self.assertAlmostEqual(mutual_information(p), np.log(2.))

    def test_fits_relation_to_KL(self):
        p = np.random.dirichlet([3.] * 10).reshape((2, 5))
        i = mutual_information(p)
        d = generalized_KL(p.flatten(),
                           (p.sum(axis=0)[np.newaxis, :]
                            * p.sum(axis=1)[:, np.newaxis]).flatten())
        self.assertAlmostEqual(i, d)


class TestConditionalEntropy(unittest.TestCase):

    def test_is_positive(self):
        p = np.random.dirichlet([3.] * 10).reshape((2, 5))
        self.assertTrue(conditional_entropy(p, axis=0) >= 0)
        self.assertTrue(conditional_entropy(p, axis=1) >= 0)

    def test_is_less_than_entropy(self):
        p = np.random.dirichlet([3.] * 10).reshape((2, 5))
        self.assertTrue(conditional_entropy(p, axis=0)
                        <= entropy(p.sum(axis=0)))
        self.assertTrue(conditional_entropy(p, axis=1)
                        <= entropy(p.sum(axis=1)))

    def test_on_independant(self):
        p = np.random.dirichlet([3.] * 10)
        p = np.vstack([.2 * p, .8 * p])
        self.assertAlmostEqual(conditional_entropy(p, axis=0),
                               entropy(p.sum(axis=0)))
        self.assertAlmostEqual(conditional_entropy(p, axis=1),
                               entropy(p.sum(axis=1)))


class TestCosineSimilarity(unittest.TestCase):

    def test_0_on_0(self):
        a = np.zeros((4, 5))
        self.assertTrue(np.all(cosine_similarity(a, a, axis=0) == 0))
        self.assertTrue(np.all(cosine_similarity(a, a, axis=1) == 0))

    def test_shapes(self):
        a = np.random.random((4, 5))
        b = np.random.random((4, 5))
        self.assertEqual(cosine_similarity(a, b).shape, (4,))
        self.assertEqual(cosine_similarity(a, b, axis=0).shape, (5,))

    def test_on_ex(self):
        a = np.vstack([np.eye(3), np.ones((1, 3))])
        b = np.array([[1., 0., 0.],
                      [1., 1., 0.],
                      [1., 0., 0.],
                      [0., 2., 1.]])
        ok = np.sqrt(np.array([1., .5, 0., 3. / 5.]))
        np.testing.assert_allclose(cosine_similarity(a, b), ok)
