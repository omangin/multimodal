from unittest import TestCase

from multimodal.lib.utils import random_split, leave_one_out


class TestRandomSplit(TestCase):

    # Default
    n = 137
    ratio = .1

    def test_returns_train_test(self):
        self.assertTrue(all([len(x) == 2
                             for x in random_split(self.n, self.ratio)]))

    def test_returns_10_sets(self):
        self.assertEqual(10, len([x for x in random_split(self.n, .1)]))

    def test_returns_9_sets(self):
        self.assertEqual(9, len([x for x in random_split(self.n, .11)]))

    def test_are_disjoint(self):
        self.assertTrue(all(
            [set(test).isdisjoint(set(train))
             for train, test in random_split(self.n, self.ratio)]))

    def test_have_all_same_size(self):
        sizes = [(len(train), len(test))
                 for train, test in random_split(self.n, self.ratio)]
        self.assertEqual(len(set(sizes)), 1)


class TestRandomSplit2(TestRandomSplit):

    n = 130


class TestLeaveOneOut(TestCase):

    # Default
    n = 37

    def test_returns_train_test(self):
        self.assertTrue(all([len(x) == 2 for x in leave_one_out(self.n)]))

    def test_returns_37_sets(self):
        self.assertEqual(37, len([x for x in leave_one_out(self.n)]))

    def test_are_disjoint(self):
        self.assertTrue(all([set(test).isdisjoint(set(train))
                             for train, test in leave_one_out(self.n)]))

    def test_have_all_same_size(self):
        sizes = [(len(train), len(test))
                 for train, test in leave_one_out(self.n)]
        self.assertEqual(len(set(sizes)), 1)
