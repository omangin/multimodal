import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from multimodal.lib.array_utils import normalize_sum, GrowingLILMatrix


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


class TestGrowingLIL(unittest.TestCase):

    def get_random_with_zeros(self, shape):
        m = np.random.random(shape)
        m *= (m < .5)
        return m

    def test_new_dimension(self):
        m = GrowingLILMatrix()
        self.assertEqual(m.shape, (0, 0))
        m.add_row(self.get_random_with_zeros((5)))
        self.assertEqual(m.shape, (1, 5))
        m.add_row(self.get_random_with_zeros((6)))
        self.assertEqual(m.shape, (2, 6))
        m.add_row(self.get_random_with_zeros((6)))
        self.assertEqual(m.shape, (3, 6))

    def test_new_values(self):
        m = GrowingLILMatrix()
        row1 = self.get_random_with_zeros((6))
        row1[5] = 0
        row1_cut = row1[:5]
        m.add_row(row1_cut)
        np.testing.assert_equal(m.todense(), np.array([row1_cut]))
        row2 = self.get_random_with_zeros((6))
        m.add_row(row2)
        np.testing.assert_equal(m.todense(), np.array([row1, row2]))

    def test_fromat(self):
        m = GrowingLILMatrix()
        self.assertEqual(m.format, 'lil')
        m.add_row(self.get_random_with_zeros((6)))
        self.assertEqual(m.format, 'lil')

    def test_to_csc(self):
        m = GrowingLILMatrix()
        m.add_row(self.get_random_with_zeros((6)))
        m.add_row(self.get_random_with_zeros((6)))
        np.testing.assert_almost_equal(m.tocsc().todense(), m.todense())
        np.testing.assert_almost_equal(m.tocsr().todense(), m.todense())
