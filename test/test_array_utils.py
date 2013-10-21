import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from multimodal.lib.array_utils import normalize_sum


class TestNormalizeSum(unittest.TestCase):

    def test_same_shape_on_1D(self):
        a = np.random.random((3,))
        norm = normalize_sum(a, axis=0)
        self.assertTrue(np.alltrue(a.shape == norm.shape))

    def test_same_shape_on_2D(self):
        a = np.random.random((2, 4))
        norm = normalize_sum(a, axis=np.random.randint(2))
        self.assertTrue(np.alltrue(a.shape == norm.shape))

    def test_same_shape_on_3D(self):
        a = np.random.random((1, 2, 3))
        norm = normalize_sum(a, axis=np.random.randint(3))
        self.assertTrue(np.alltrue(a.shape == norm.shape))

    def test_correct_on_1D(self):
        a = np.random.random((5,))
        norm = normalize_sum(a, axis=0)
        assert_array_almost_equal(1., np.sum(norm))

    def test_correct_on_2D_axis0(self):
        a = np.array([[0., 1., 3.], [2., 3., 3.]])
        norm = normalize_sum(a, axis=0)
        ok = np.array([[0., .25, .5], [1., .75, .5]])
        self.assertTrue(np.alltrue(norm == ok))

    def test_correct_on_2D_axis1(self):
        a = np.array([[0., 1., 3.], [2., 3., 3.]])
        norm = normalize_sum(a, axis=1)
        ok = np.array([[0., .25, .75], [.25, .375, .375]])
        self.assertTrue(np.alltrue(norm == ok))

    def test_correct_on_3D(self):
        a = np.random.random((2, 4, 5))
        ax = np.random.randint(3)
        norm = normalize_sum(a, axis=ax)
        assert_array_almost_equal(np.sum(norm, ax), 1.)

    def test_error_on_wrong_axis(self):
        a = np.random.random((2, 3, 4))
        with self.assertRaises(ValueError):
            normalize_sum(a, axis=3)

    def test_robust_on_zero(self):
        a = np.random.random((2, 4))
        a[1, :] *= 0
        norm = normalize_sum(a, axis=1)
        assert(not np.any(np.isnan(norm)))
