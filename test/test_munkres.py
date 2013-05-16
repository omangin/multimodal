import unittest

import numpy as np

from multimodal.lib.munkres import (min_weight_perm,
                                    min_weight_perm_brute_force,
                                    weight)


class TestMinWeightPerm(unittest.TestCase):

    def test1(self):
        weights = np.array([[0, 1, 10], [1, 5, 10], [10, 10, 0]])
        found_perm = min_weight_perm(weights)
        found_w = weight(found_perm, weights)
        self.assertEqual(found_perm, [1, 0, 2])
        self.assertEqual(found_w, 2)

    def test2(self):
        weights = np.random.random((7, 7))
        found_HM = min_weight_perm(weights)
        found_BF = min_weight_perm_brute_force(weights)
        self.assertEqual(found_HM, found_BF)

    def test_on_non_square(self):
        weights = np.array([[0, 1, 10, 20], [1, 5, 10, 20], [10, 10, 0, 20]])
        found_perm = min_weight_perm(weights)
        self.assertEqual(found_perm, [1, 0, 2])

    def test_on_non_square2(self):
        weights = np.array([[0, 1, 10, 20], [1, 5, 10, 20], [10, 10, 0, 20]]).T
        found_perm = min_weight_perm(weights)
        self.assertEqual(found_perm, [1, 0, 2, None])
