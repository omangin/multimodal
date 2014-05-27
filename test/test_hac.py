import unittest

import numpy as np

from multimodal.features.hac import coocurrences


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
