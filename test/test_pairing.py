from unittest import TestCase, skip
import random

from multimodal.lib.window import BasicSlidingWindow, ConcatSlidingWindow
from multimodal.pairing import (associate, flatten, organize_by_values,
                                associate_samples, associate_to_window)


class TestVar(TestCase):

    def test_associate(self):
        l = [[[1, 2], [3, 4]],
             [['a', 'b', 'c'], [',', '?']],
             [[0.3, 0.4], [0.4, 0.2]],
             ]
        assoc = [[(1, 'a', .3), (2, 'b', .4)], [(3, ',', .4), (4, '?', .2)]]
        self.assertEqual(associate(l), assoc)

    def test_flatten(self):
        l = [range(4), range(4, 10), range(10, 15)]
        self.assertEqual(flatten(l), range(15))

    def test_organize_by_values(self):
        l = [0, 0, 1, 1, 0, 2, 1]
        org = [[0, 1, 4], [2, 3, 6], [5]]
        self.assertEqual(organize_by_values(l, 3), org)


class TestModalityAssociation(TestCase):

    def setUp(self):
        self.sets = [[1, 1, 2, 1, 3, 2, 3], ['4', '5', '6', '5', '5', '4']]

    def test_fails_on_different_nb_of_labels(self):
        self.sets[0].append(0)
        with self.assertRaises(AssertionError):
            associate_samples(self.sets)

    def test_names(self):
        names, _dum, _my = associate_samples(self.sets)
        self.assertEqual(set(names[0]), set(range(1, 4)))
        self.assertEqual(set(names[1]), set(['4', '5', '6']))

    def test_labels_match_between(self):
        names, _dummy, assocs = associate_samples(self.sets)
        self.assertEqual([names[0].index(self.sets[0][x[0]]) for x in assocs],
                         [names[1].index(self.sets[1][x[1]]) for x in assocs])

    def test_labels_match_origin(self):
        names, labels, assocs = associate_samples(self.sets)
        self.assertEqual([names[0][l] for l in labels],
                         [self.sets[0][x[0]] for x in assocs])
        self.assertEqual([names[1][l] for l in labels],
                         [self.sets[1][x[1]] for x in assocs])

    def test_labels_shuffled(self):
        random.seed(0)
        names, labels, assocs = associate_samples(self.sets, shuffle=True)
        self.assertEqual([names[0][l] for l in labels],
                         [self.sets[0][x[0]] for x in assocs])
        self.assertEqual([names[1][l] for l in labels],
                         [self.sets[1][x[1]] for x in assocs])
        self.assertEquals(names, ([1, 2, 3], ['6', '5', '4']))

    @skip("Too long to run each time but good to have!")
    def test_labels_match_on_db(self):
        from multimodal.experiment import AcornsLoader, Choreo2Loader
        speaker = 1
        sound_loader = AcornsLoader(speaker)
        motion_loader = Choreo2Loader()
        sound_labels = sound_loader.get_labels()
        motion_labels = motion_loader.get_labels()
        raw_labels = [loader.get_labels() for loader in [sound_loader,
                                                         motion_loader]]
        names, labels, tuples = associate_samples(raw_labels)
        sound_names = names[0]
        sound_idx = [t[0] for t in tuples]
        motion_names = names[1]
        motion_idx = [t[1] for t in tuples]
        # Check sound output labels
        self.assertEqual([sound_names[l] for l in labels],
                         [sound_labels[i] for i in sound_idx])
        # Check motion output labels
        self.assertEqual([motion_names[l] for l in labels],
                         [motion_labels[i] for i in motion_idx])


def get_win(labels, times):
    wins = [BasicSlidingWindow(0., t, obj=l) for l, t in zip(labels, times)]
    return ConcatSlidingWindow(ConcatSlidingWindow.align(wins))


class TestWindowedAssociation(TestCase):

    def setUp(self):
        self.sets = [[0, 1, 0], [0, 1, 1, 0, 0, 1]]

    def test_fails_on_missing_frame(self):
        with self.assertRaises(ValueError):
            associate_to_window(get_win(self.sets[0], [7.2, 3.1, 1.05]),
                                self.sets[1], range(6), 1.)

    def test_associate_to_windows(self):
        win = associate_to_window(get_win(self.sets[0], [1.8, 3.1, 1.05]),
                                  range(6), self.sets[1], 1.)
        self.assertEquals([self.sets[1][w.obj] for w in win.windows],
                          [0, 0, 1, 1, 1, 0])

    def test_associate_to_windows_reverse_indices(self):
        win = associate_to_window(get_win(self.sets[0], [1.8, 3.1, 1.05]),
                                  range(6)[::-1], self.sets[1][::-1], 1.)
        self.assertEquals([self.sets[1][w.obj] for w in win.windows],
                          [0, 0, 1, 1, 1, 0])
