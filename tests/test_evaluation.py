import unittest

import numpy as np
import scipy.sparse as sp

from multimodal.lib.array_utils import normalize_features
from multimodal.evaluation import (evaluate_label_reco,
                                   evaluate_NN_label,
                                   chose_examples)


class TestLabelEvaluation(unittest.TestCase):

    def test(self):
        labels = [2, 0]
        reco = np.array([[.1, .5, .6, .1],
                         [.6, .5, .2, .1]])
        good = evaluate_label_reco(reco, labels)
        self.assertEqual(good, 1.)
        bad = evaluate_label_reco(reco[[1, 0], :], labels)
        self.assertEqual(bad, 0.)
        medium = evaluate_label_reco(reco[[1, 1], :], labels)
        self.assertEqual(medium, .5)

    def test_fails_on_multiple_labels(self):
        labels = [[2], [0]]
        reco = np.array([[.1, .5, .6, .1],
                         [.6, .5, .2, .1]])
        with self.assertRaises(AssertionError):
            evaluate_label_reco(reco, labels)


class TestNNEvaluation(unittest.TestCase):

    def setUp(self):
        self.labels_a = np.random.randint(10, size=13)
        self.labels_b = [i for i in reversed(range(10))]
        # Encode label on third coordinate of a and fourth of b
        self.a = np.random.random((13, 5))
        for i in range(13):
            self.a[i, 2] = self.labels_a[i]
        self.b = np.random.random((10, 5))
        for i in range(10):
            self.b[i, 3] = self.labels_b[i]

    def fake_metrics(self, a, b, axis=-1):
        assert(axis == -1)  # Test does not work if not...
        return 1. - (a[:, :, 2] == b[:, :, 3])

    def test_good_on_fake_measure(self):
        self.assertEqual(evaluate_NN_label(self.a, self.b, self.labels_a,
                                           self.labels_b, self.fake_metrics
                                           ), 1.)

    def test_bad_on_fake_measure(self):
        self.assertEqual(evaluate_NN_label(self.a, 1 + self.b,
                                           self.labels_a, self.labels_b,
                                           self.fake_metrics), 0.)

    def test_on_fake_measure_sparse(self):
        a = sp.lil_matrix(self.a).tocsr()
        b = sp.lil_matrix(self.b).tocsr()
        self.assertEqual(
                evaluate_NN_label(a, b, self.labels_a, self.labels_b,
                                  self.fake_metrics),
                1.)
        self.assertEqual(
                evaluate_NN_label(a, 1 + self.b, self.labels_a,
                                  self.labels_b, self.fake_metrics),
                0.)


class TestChoseExamples(unittest.TestCase):

    def setUp(self):
        self.label_set = list(range(3))
        self.labels = self.label_set * 5
        np.random.seed(0)
        np.random.shuffle(self.labels)

    def test_choses_as_many_examples_as_labels(self):
        r = chose_examples(self.labels, self.label_set)
        self.assertEqual(len(r), len(self.label_set))
        r = chose_examples(self.labels)  # And without giving labels
        self.assertEqual(len(r), len(self.label_set))

    def test_choses_twice_as_many_examples_as_labels(self):
        r = chose_examples(self.labels, self.label_set, number=2)
        self.assertEqual(len(r), 2 * len(self.label_set))

    def test_all_chosen_are_indices(self):
        r = chose_examples(self.labels, self.label_set, number=2)
        assert(all([0 <= i < len(self.labels) for i in r]))

    def test_all_labels_are_chosen_once(self):
        r = chose_examples(self.labels, self.label_set)
        lab = [self.labels[i] for i in r]
        assert(all([lab.count(l) == 1 for l in self.label_set]))

    def test_all_labels_are_chosen_twice(self):
        r = chose_examples(self.labels, self.label_set, number=2)
        lab = [self.labels[i] for i in r]
        assert(all([lab.count(l) == 2 for l in self.label_set]))


class TestNormalizeFeatures(unittest.TestCase):

    def setUp(self):
        self.mat = np.random.random((32, 13))
        self.mat = 10. * self.mat * (self.mat < .2)

    def test_on_sparse_same_shape(self):
        m = self.mat
        m[0, :] += 1  # Ensures that no column has zero sum
        m = sp.csc_matrix(m)
        norm = normalize_features(m)
        assert(np.allclose(norm.sum(axis=0), 1))
        self.assertEqual(norm.shape, m.shape)

    def test_removes_columns_sparse(self):
        m = self.mat
        m[0, :] += 1  # Ensures that no column has zero sum
        m[:, [1, 3]] = 0  # Ensures column 1 and 3 have zero sum
        m = sp.csc_matrix(m)
        norm = normalize_features(m)
        assert(np.allclose(norm.sum(axis=0), 1))
        self.assertEqual(norm.shape, (m.shape[0], m.shape[1] - 2))

    def test_on_dense_same_shape(self):
        m = self.mat
        m[0, :] += 1  # Ensures that no column has zero sum
        norm = normalize_features(m)
        assert(np.allclose(norm.sum(axis=0), 1))
        self.assertEqual(norm.shape, m.shape)

    def test_removes_columns_dense(self):
        m = self.mat
        m[0, :] += 1  # Ensures that no column has zero sum
        m[:, [1, 3]] = 0  # Ensures column 1 and 3 have zero sum
        norm = normalize_features(m)
        assert(np.allclose(norm.sum(axis=0), 1))
        self.assertEqual(norm.shape, (m.shape[0], m.shape[1] - 2))

    def test_same_on_dense_and_sparse(self):
        m1 = sp.csc_matrix(self.mat)
        m2 = sp.csr_matrix(self.mat)
        n = normalize_features(self.mat)
        n1 = normalize_features(m1)
        n2 = normalize_features(m2)
        assert(np.allclose(n1.todense(), n))
        assert(np.allclose(n2.todense(), n))

    def test_does_not_modify(self):
        m = self.mat.copy()
        normalize_features(m)
        ms = sp.csr_matrix(m)
        normalize_features(ms)
        assert(np.allclose(m, self.mat))
        assert(np.allclose(ms.todense(), self.mat))

    def test_OK(self):
        n = normalize_features(np.array([[1., 0., 1.5, .1],
                                         [1., 0.,  .5, .3]]))
        ok = np.array([[.5, .75, .25],
                       [.5, .25, .75]])
        assert(np.allclose(n, ok))
