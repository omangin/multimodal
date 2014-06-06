import unittest

import numpy as np
from scipy.cluster.vq import kmeans2

from multimodal.features.hac import coocurrences, iterative_kmeans


class TestCoocurrences(unittest.TestCase):

    def setUp(self):
        self.n_centro = 13
        self.random_quantized = np.random.randint(self.n_centro, size=(100))

    def test_shape_is_square(self):
        result = coocurrences(self.random_quantized, self.n_centro, lag=17)
        self.assertEquals(result.shape, (self.n_centro ** 2,))

    def test_count_is_same(self):
        result = coocurrences(self.random_quantized, self.n_centro, lag=17)
        self.assertEquals(sum(result), self.random_quantized.shape[0] - 17)

    def test_dummy(self):
        quantized = np.array([0, 1, 1, 0, 1, 1, 1])
        result = coocurrences(quantized, 2, lag=2)
        counts = [0, 1, 2, 2]
        self.assertEquals(list(result), counts)


class TestIterativeKmeans(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.cent = [np.array([0, 1]), np.array([0, -1]), np.array([10, 0])]
        self.data = np.vstack([
            np.random.multivariate_normal(c, .5 * np.eye(2), 10)
            for c in self.cent])
        np.random.shuffle(self.data)

    def test_shape_is_ok(self):
        data = np.random.random((20, 3))
        centroids = iterative_kmeans(data, 5)
        self.assertEquals(centroids.shape, (5, 3))

    def test_is_converged(self):
        data = np.random.random((20, 3))
        centroids = iterative_kmeans(data, 5)
        centroids2, _ = kmeans2(data, centroids, minit='matrix')
        np.testing.assert_allclose(centroids, centroids2)

    def test_ex_k2(self):
        centroids = iterative_kmeans(self.data, 2)
        self.assertTrue(
            np.allclose(centroids, np.array([[0, 0], [10, 0]]), atol=1)
            or np.allclose(centroids, np.array([[10, 0], [0, 0]]), atol=1)
            )

    def test_ex_k3(self):
        centroids = iterative_kmeans(self.data, 3)
        self.assertTrue(
            np.allclose(centroids,
                        np.array([[0, 1], [0, -1], [10, 0]]), atol=1)
            or np.allclose(centroids,
                           np.array([[0, 1], [0, -1], [10, 0]]), atol=1)
            or np.allclose(centroids,
                           np.array([[10, 0], [0, -1], [0, 1]]), atol=1)
            or np.allclose(centroids,
                           np.array([[10, 0], [0, 1], [0, -1]]), atol=1)
            )
