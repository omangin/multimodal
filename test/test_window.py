import os
from unittest import TestCase
import tempfile
import shutil

import numpy as np
from scipy.io import wavfile

from multimodal.lib.window import (
        _to_approx_int,
        TimeOutOfBound,
        BasicSlidingWindow,
        ArraySlidingWindow,
        WavFileSlidingWindow,
        ConcatSlidingWindow,
        concat_from_list_of_wavs,
        )


WAV_RATE = 44100
SAMPLE_DELTA = 1. / WAV_RATE


class TestToApproxInt(TestCase):

    tol = .1

    def test_is_int(self):
        self.assertIsInstance(_to_approx_int(3.14), int)

    def test_is_id_for_int(self):
        l = range(-10, 10)
        self.assertEqual([_to_approx_int(i) for i in l], l)

    def test_is_below(self):
        self.assertEqual(_to_approx_int(3.14, tol=self.tol), 3)
        self.assertEqual(_to_approx_int(3.14, tol=self.tol, above=False), 3)
        self.assertEqual(_to_approx_int(-3.14, tol=self.tol), -4)

    def test_is_above(self):
        self.assertEqual(_to_approx_int(3.14, tol=self.tol, above=True), 4)
        self.assertEqual(_to_approx_int(-3.14, tol=self.tol, above=True), -3)

    def test_is_closest(self):
        self.assertEqual(_to_approx_int(3.94, tol=self.tol), 4)
        self.assertEqual(_to_approx_int(3.94, tol=self.tol, above=False), 4)
        self.assertEqual(_to_approx_int(3.04, tol=self.tol, above=True), 3)
        self.assertEqual(_to_approx_int(-3.04, tol=self.tol), -3)
        self.assertEqual(_to_approx_int(-3.94, tol=self.tol, above=True), -4)


class AbstractTestSlidingWindow(object):

    def test_start_end(self):
        self.assertAlmostEqual(self.win.absolute_start, 2.)
        self.assertAlmostEqual(self.win.absolute_end, 4.)

    def test_get_subwindow_raises_TimeOutOfBound(self):
        with self.assertRaises(TimeOutOfBound):
            self.win.get_subwindow(0.,  3.)
        with self.assertRaises(TimeOutOfBound):
            self.win.get_subwindow(3.,  5.)

    def test_get_subwindow_times(self):
        subwin = self.win.get_subwindow(2.5, 3)
        self.assertAlmostEquals(subwin.absolute_start, 2.5)
        self.assertAlmostEquals(subwin.absolute_end, 3.)

    def test_get_subwindow_returns_empty(self):
        subwin = self.win.get_subwindow(3.5, 2.5)
        self.assertEquals(subwin.duration(), 0)


class TestBasicSlidingWindow(AbstractTestSlidingWindow, TestCase):

    def setUp(self):
        self.win = BasicSlidingWindow(2., 4., obj='zou')


class TestArraySlidingWindow(AbstractTestSlidingWindow, TestCase):

    def setUp(self):
        self.win = ArraySlidingWindow(range(20, 10, -1), 2., 5.)

    def test_get_subwindow_times(self):
        subwin = self.win.get_subwindow(2.5, 3)
        self.assertAlmostEquals(subwin.absolute_start, 2.6)
        self.assertAlmostEquals(subwin.absolute_end, 3.)
        subwin = self.win.get_subwindow(2.4, 3)
        self.assertAlmostEquals(subwin.absolute_start, 2.4)
        self.assertAlmostEquals(subwin.absolute_end, 3.)
        subwin = self.win.get_subwindow(2.4, 3.2)
        self.assertAlmostEquals(subwin.absolute_start, 2.4)
        self.assertAlmostEquals(subwin.absolute_end, 3.2)

    def test_get_subwindow_ok(self):
        subwin = self.win.get_subwindow(2.5, 3)
        self.assertEquals(subwin.array, range(17, 15, -1))
        subwin = self.win.get_subwindow(2.4, 3)
        self.assertEquals(subwin.array, range(18, 15, -1))
        subwin = self.win.get_subwindow(2.4, 3.2)
        self.assertEquals(subwin.array, range(18, 14, -1))

    def test_concatenate(self):
        win2 = ArraySlidingWindow(range(5), 4., 5.)
        win = ArraySlidingWindow.concatenate([self.win, win2])
        self.assertEquals(win.absolute_start, 2.)
        self.assertEquals(win.absolute_end, 5.)
        self.assertEquals(win.duration(), 3.)
        np.testing.assert_array_equal(win.array,
                                      np.array(range(20, 10, -1) + range(5)))


class WavTestCase(TestCase):

    def setUp(self):
        test_set = [i * np.ones(WAV_RATE * i) for i in range(1, 5)]
        self.tmpdir = tempfile.mkdtemp()
        self.files = [os.path.join(self.tmpdir, "wav%s" % i)
                      for i in range(1, 5)]
        for (wave, filename) in zip(test_set, self.files):
            wavfile.write(filename, WAV_RATE, np.int16(wave))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)


class TestWavFileSlidingWindow(WavTestCase):

    def test_create(self):
        win = WavFileSlidingWindow(self.files[0], 3.14)
        self.assertAlmostEqual(win.absolute_start, 3.14, delta=SAMPLE_DELTA)
        self.assertAlmostEqual(win.absolute_end, 4.14, delta=SAMPLE_DELTA)
        self.assertAlmostEqual(win.duration(), 1., delta=SAMPLE_DELTA)

    def test_sub_window(self):
        win = WavFileSlidingWindow(self.files[0], 3.14)
        delay_start = 500 * SAMPLE_DELTA
        new_start = 3.14 + delay_start
        delay_stop = 1000 * SAMPLE_DELTA
        new_stop = 3.14 + delay_stop + .5 * SAMPLE_DELTA
        subwin = win.get_subwindow(new_start, new_stop)
        self.assertAlmostEqual(subwin.absolute_start, new_start)
        self.assertAlmostEqual(subwin.absolute_end, 3.14 + delay_stop)
        self.assertAlmostEqual(subwin.duration(), 500 * SAMPLE_DELTA)
        self.assertEqual(subwin.n_samples, 500)

    def test_sub_sub_window(self):
        win = WavFileSlidingWindow(self.files[0], 3.14)
        new_start = 3.14 + 500 * SAMPLE_DELTA
        new_stop = 3.14 + 1000.5 * SAMPLE_DELTA
        subwin = win.get_subwindow(new_start, new_stop)
        new_new_start = new_start + 111 * SAMPLE_DELTA
        new_new_stop = new_new_start + 222 * SAMPLE_DELTA
        subsubwin = subwin.get_subwindow(new_new_start - .5 * SAMPLE_DELTA,
                                      new_new_stop)
        self.assertAlmostEqual(subsubwin.absolute_start, new_new_start)
        self.assertAlmostEqual(subsubwin.absolute_end,
                               3.14 + (500 + 333) * SAMPLE_DELTA)
        self.assertAlmostEqual(subsubwin.duration(), 222 * SAMPLE_DELTA)
        self.assertEqual(subsubwin.n_samples, 222)

    def test_to_array_window(self):
        win = WavFileSlidingWindow(self.files[0], 3.14)
        values = np.ones(WAV_RATE)
        array_win = win.to_array_window()
        np.testing.assert_array_equal(array_win.array, values)

    def test_to_array_sub_window(self):
        win = WavFileSlidingWindow(self.files[0], 3.14)
        subwin = win.get_subwindow(3.14, 3.64)  # assumes 2 devides WAV_RATE
        values = np.ones(.5 * WAV_RATE)
        array_win = subwin.to_array_window()
        np.testing.assert_array_equal(array_win.array, values)


class TestConcatSlidingWindow(TestCase):

    def setUp(self):
        self.wins = [ArraySlidingWindow(range(20, 10, -1), 2., 5),  # 2s
                     ArraySlidingWindow(range(10), 4., 10),  # 1s
                     ArraySlidingWindow(range(20), 5., 10),  # 2s
                     ]
        self.win = ConcatSlidingWindow(self.wins)

    def test_time(self):
        self.assertAlmostEqual(self.win.absolute_start, 2.)
        self.assertAlmostEqual(self.win.absolute_end, 7.)

    def test_get_file_index_from_time(self):
        with self.assertRaises(TimeOutOfBound):
            self.win._get_file_index_from_time(9.)
        self.assertEquals(self.win._get_file_index_from_time(
            self.win.absolute_end - 0.001),
            len(self.wins) - 1)

    def test_raises_ValueError_on_non_consecutive(self):
        self.wins[1].move(-1.)
        with self.assertRaises(ValueError):
            ConcatSlidingWindow(self.wins)

    def test_absolute_bounds(self):
        self.assertAlmostEquals(self.win.absolute_start, 2.)
        self.assertAlmostEquals(self.win.absolute_end, 7.)

    def test_get_subwindow(self):
            # all
            subwin = self.win.get_subwindow(2.7, 5.4)
            self.assertAlmostEqual(subwin.absolute_start, 2.8)
            self.assertAlmostEqual(subwin.absolute_end, 5.4)
            self.assertEqual(len(subwin.windows), 3)
            # first
            subwin = self.win.get_subwindow(2.2, 3.1)
            self.assertAlmostEqual(subwin.absolute_start, 2.2)
            self.assertAlmostEqual(subwin.absolute_end, 3.0)
            self.assertEqual(len(subwin.windows), 1)
            # last
            subwin = self.win.get_subwindow(5.1, 6.1)
            self.assertAlmostEqual(subwin.absolute_start, 5.1)
            self.assertAlmostEqual(subwin.absolute_end, 6.1)
            self.assertEqual(len(subwin.windows), 1)
            # to end
            subwin = self.win.get_subwindow(2.2, 7.0)
            self.assertAlmostEqual(subwin.absolute_start, 2.2)
            self.assertAlmostEqual(subwin.absolute_end, 7.0)
            self.assertEqual(len(subwin.windows), 3)
            # to end of second
            subwin = self.win.get_subwindow(2.2, 5.0)
            self.assertAlmostEqual(subwin.absolute_start, 2.2)
            self.assertAlmostEqual(subwin.absolute_end, 5.0)
            self.assertEqual(len(subwin.windows), 2)
            # from start
            subwin = self.win.get_subwindow(2.0, 4.2)
            self.assertAlmostEqual(subwin.absolute_start, 2.0)
            self.assertAlmostEqual(subwin.absolute_end, 4.2)
            self.assertEqual(len(subwin.windows), 2)
            # from start of second
            subwin = self.win.get_subwindow(4.0, 4.2)
            self.assertAlmostEqual(subwin.absolute_start, 4.0)
            self.assertAlmostEqual(subwin.absolute_end, 4.2)
            self.assertEqual(len(subwin.windows), 2)

    def test_to_array_window(self):
        win = ConcatSlidingWindow(self.wins[1:])
        values = np.array(range(10) + range(20))
        subwin = win.get_subwindow(4., 7.)
        np.testing.assert_array_equal(win.to_array_window().array, values)
        np.testing.assert_array_equal(subwin.to_array_window().array, values)

    def test_to_array_sub_window(self):
        subwin = self.win.get_subwindow(4.79, 5.49)
        values = np.array(range(8, 10) + range(4))
        np.testing.assert_array_equal(subwin.to_array_window().array, values)


class TestConcatWavFileSlidingWindow(WavTestCase):

    def setUp(self):
        WavTestCase.setUp(self)
        self.win = concat_from_list_of_wavs(self.files)

    def test_wav_load(self):
        self.assertEquals(self.win.absolute_end, self.win.absolute_start + 10)
        self.assertEquals(self.win.windows[0].rate, WAV_RATE)

    def test_get_file_index_from_time(self):
        with self.assertRaises(TimeOutOfBound):
            self.win._get_file_index_from_time(self.win.absolute_start + 11)
        self.assertEquals(self.win._get_file_index_from_time(
            self.win.absolute_end - 0.001),
            len(self.files) - 1)

    def test_get_subwindow(self):
        win = self.win.get_subwindow(0.5, 0.7)
        np.testing.assert_array_equal(win._ends(), [.7])
        self.assertEqual(win.absolute_end, .7)
        win = self.win.get_subwindow(0.5, 5.3)
        np.testing.assert_array_equal(win._ends(), [1, 3, 5.3])
        self.assertEqual(win.absolute_end, 5.3)

    def test_to_array_window(self):
        values = np.concatenate([
            np.ones(WAV_RATE),
            2 * np.ones(WAV_RATE * 2),
            3 * np.ones(WAV_RATE * 3),
            4 * np.ones(4 * WAV_RATE)])
        array_win = [elem for elem in self.win.to_array_window().array]
        np.testing.assert_array_equal(array_win, values)

    def test_to_array_sub_window(self):
        win = self.win.get_subwindow(.1, 7.3)
        all_values = np.concatenate([
            np.ones(WAV_RATE),
            2 * np.ones(WAV_RATE * 2),
            3 * np.ones(WAV_RATE * 3),
            4 * np.ones(4 * WAV_RATE)])
        time = np.array(range(10 * WAV_RATE))
        sub_values = all_values[np.nonzero((time >= .1 * WAV_RATE)
                                            * (time + 1 <= 7.3 * WAV_RATE))]
        np.testing.assert_array_equal(win.to_array_window().array, sub_values)
