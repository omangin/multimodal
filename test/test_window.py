import os
from unittest import TestCase
import tempfile
import shutil

import numpy as np
from scipy.io import wavfile

from multimodal.lib.window import (
    _to_approx_int,
    TimeOutOfBound,
    BasicTimeWindow,
    ArrayTimeWindow,
    WavFileTimeWindow,
    slider,
    ConcatTimeWindow,
    concat_from_list_of_wavs,
    concat_of_frames,
    )


WAV_RATE = 44100
SAMPLE_DELTA = 1. / WAV_RATE


class TestToApproxInt(TestCase):

    tol = .1

    def test_is_int(self):
        self.assertIsInstance(_to_approx_int(3.14), int)

    def test_is_id_for_int(self):
        l = [x for x in range(-10, 10)]
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


class AbstractTestTimeWindow(object):

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
        self.assertAlmostEqual(subwin.absolute_start, 2.5)
        self.assertAlmostEqual(subwin.absolute_end, 3.)

    def test_get_subwindow_returns_empty(self):
        subwin = self.win.get_subwindow(3.5, 2.5)
        self.assertEqual(subwin.duration(), 0)


class TestBasicTimeWindow(AbstractTestTimeWindow, TestCase):

    def setUp(self):
        self.win = BasicTimeWindow(2., 4., obj='zou')


class TestArrayTimeWindow(AbstractTestTimeWindow, TestCase):

    def setUp(self):
        self.win = ArrayTimeWindow(range(20, 10, -1), 2., 5.)

    def test_get_subwindow_times(self):
        subwin = self.win.get_subwindow(2.5, 3)
        self.assertAlmostEqual(subwin.absolute_start, 2.6)
        self.assertAlmostEqual(subwin.absolute_end, 3.)
        subwin = self.win.get_subwindow(2.4, 3)
        self.assertAlmostEqual(subwin.absolute_start, 2.4)
        self.assertAlmostEqual(subwin.absolute_end, 3.)
        subwin = self.win.get_subwindow(2.4, 3.2)
        self.assertAlmostEqual(subwin.absolute_start, 2.4)
        self.assertAlmostEqual(subwin.absolute_end, 3.2)

    def test_get_subwindow_ok(self):
        subwin = self.win.get_subwindow(2.5, 3)
        self.assertEqual(subwin.array, range(17, 15, -1))
        subwin = self.win.get_subwindow(2.4, 3)
        self.assertEqual(subwin.array, range(18, 15, -1))
        subwin = self.win.get_subwindow(2.4, 3.2)
        self.assertEqual(subwin.array, range(18, 14, -1))

    def test_concatenate(self):
        win2 = ArrayTimeWindow(range(5), 4., 5.)
        win = ArrayTimeWindow.concatenate([self.win, win2])
        self.assertEqual(win.absolute_start, 2.)
        self.assertEqual(win.absolute_end, 5.)
        self.assertEqual(win.duration(), 3.)
        np.testing.assert_array_equal(win.array,
                                      np.array(list(range(20, 10, -1)) +
                                               list(range(5))))


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


class TestWavFileTimeWindow(WavTestCase):

    def test_create(self):
        win = WavFileTimeWindow(self.files[0], 3.14)
        self.assertAlmostEqual(win.absolute_start, 3.14, delta=SAMPLE_DELTA)
        self.assertAlmostEqual(win.absolute_end, 4.14, delta=SAMPLE_DELTA)
        self.assertAlmostEqual(win.duration(), 1., delta=SAMPLE_DELTA)

    def test_sub_window(self):
        win = WavFileTimeWindow(self.files[0], 3.14)
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
        win = WavFileTimeWindow(self.files[0], 3.14)
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

    def test_copy_sub_window(self):
        win = WavFileTimeWindow(self.files[0], 3.14)
        new_start = 3.14 + 500 * SAMPLE_DELTA
        new_stop = 3.14 + 1000.5 * SAMPLE_DELTA
        subwin = win.get_subwindow(new_start, new_stop)
        copy = subwin.copy()
        self.assertAlmostEqual(copy.absolute_start, subwin.absolute_start)
        self.assertAlmostEqual(copy.absolute_end, subwin.absolute_end)
        self.assertEqual(copy.n_samples, subwin.n_samples)

    def test_to_array_window(self):
        win = WavFileTimeWindow(self.files[0], 3.14)
        values = np.ones(WAV_RATE)
        array_win = win.to_array_window()
        np.testing.assert_array_equal(array_win.array, values)

    def test_to_array_sub_window(self):
        win = WavFileTimeWindow(self.files[0], 3.14)
        subwin = win.get_subwindow(3.14, 3.64)  # assumes 2 devides WAV_RATE
        values = np.ones(.5 * WAV_RATE)
        array_win = subwin.to_array_window()
        np.testing.assert_array_equal(array_win.array, values)


class TestConcatTimeWindow(TestCase):

    def setUp(self):
        self.wins = [ArrayTimeWindow(range(20, 10, -1), 2., 5),  # 2s
                     ArrayTimeWindow(range(10), 4., 10),  # 1s
                     ArrayTimeWindow(range(20), 5., 10),  # 2s
                     ]
        self.win = ConcatTimeWindow(self.wins)

    def test_time(self):
        self.assertAlmostEqual(self.win.absolute_start, 2.)
        self.assertAlmostEqual(self.win.absolute_end, 7.)

    def test_get_file_index_from_time(self):
        with self.assertRaises(TimeOutOfBound):
            self.win._get_file_index_from_time(9.)
        self.assertEqual(self.win._get_file_index_from_time(
            self.win.absolute_end - 0.001),
            len(self.wins) - 1)

    def test_raises_ValueError_on_non_consecutive(self):
        self.wins[1].move(-1.)
        with self.assertRaises(ValueError):
            ConcatTimeWindow(self.wins)

    def test_absolute_bounds(self):
        self.assertAlmostEqual(self.win.absolute_start, 2.)
        self.assertAlmostEqual(self.win.absolute_end, 7.)

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
        win = ConcatTimeWindow(self.wins[1:])
        values = np.array(list(range(10)) + list(range(20)))
        subwin = win.get_subwindow(4., 7.)
        np.testing.assert_array_equal(win.to_array_window().array, values)
        np.testing.assert_array_equal(subwin.to_array_window().array, values)

    def test_to_array_sub_window(self):
        subwin = self.win.get_subwindow(4.79, 5.49)
        values = np.array(list(range(8, 10)) + list(range(4)))
        np.testing.assert_array_equal(subwin.to_array_window().array, values)


class TestConcatWavFileTimeWindow(WavTestCase):

    def setUp(self):
        WavTestCase.setUp(self)
        self.win = concat_from_list_of_wavs(self.files)

    def test_wav_load(self):
        self.assertEqual(self.win.absolute_end, self.win.absolute_start + 10)
        self.assertEqual(self.win.windows[0].rate, WAV_RATE)

    def test_get_file_index_from_time(self):
        with self.assertRaises(TimeOutOfBound):
            self.win._get_file_index_from_time(self.win.absolute_start + 11)
        self.assertEqual(self.win._get_file_index_from_time(
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
        sub_values = all_values[np.nonzero((time >= .1 * WAV_RATE) *
                                           (time + 1 <= 7.3 * WAV_RATE))]
        np.testing.assert_array_equal(win.to_array_window().array, sub_values)


class Testslider(TestCase):

    def test_simple_frames(self):
        slider1 = slider(1., 5., 2., 2.)
        slider2 = slider(1., 5., 2., 2., partial=True)
        ok = [(1., 3.), (3., 5.)]
        self.assertEqual(slider1, ok)
        self.assertEqual(slider2, ok)

    def test_frames(self):
        slider1 = slider(0., 5., 2., 2.)
        slider2 = slider(0., 5., 2., 2., partial=True)
        ok = [(0., 2.), (2., 4.)]
        self.assertEqual(slider1, ok)
        self.assertEqual(slider2, ok + [(4., 5.)])

    def test_simple_overlap(self):
        slider1 = slider(1., 5., 2., 2. / 3.)
        slider2 = slider(1., 5., 2., 2. / 3., partial=True)
        ok = [(x / 3., y / 3.)
              for x, y in [(3., 9.), (5., 11.), (7., 13.), (9., 15.)]]
        np.testing.assert_allclose(slider1, ok)
        np.testing.assert_allclose(slider2,
                                   ok + [(11. / 3., 5.), (13. / 3., 5.)])

    def test_overlap(self):
        slider1 = slider(0., 2., 1., 2. / 3.)
        slider2 = slider(0., 2., 1., 2. / 3., partial=True)
        ok = [(x / 3., y / 3.) for x, y in [(0., 3.), (2., 5.)]]
        np.testing.assert_allclose(slider1, ok)
        np.testing.assert_allclose(slider2, ok + [(4. / 3., 2.)])

    def test_overlap_several_partials(self):
        slider1 = slider(0., 1., 2., .4)
        slider2 = slider(0., 1., 2., .4, partial=True)
        np.testing.assert_allclose(slider1, [])
        np.testing.assert_allclose(slider2, [(0., 1.), (.4, 1.), (.8, 1.)])


class TestConcatOfFrames(TestCase):

    t_start = 1.3
    t_end = 13.
    rate = 3

    def old_concat_of_frames(self):
        """Old implementation to test against."""
        n_frames = _to_approx_int((self.t_end - self.t_start) * self.rate,
                                  above=True)
        duration = 1. / self.rate
        return ConcatTimeWindow(
            ConcatTimeWindow.align([BasicTimeWindow(0., duration)
                                    for _dummy in range(n_frames)],
                                   start=self.t_start)
            ).get_subwindow(self.t_start, self.t_end)

    def test_is_ConcatTimeWindow(self):
        win = concat_of_frames(self.t_start, self.t_end, self.rate)
        self.assertIsInstance(win, ConcatTimeWindow)

    def test_duration(self):
        win = concat_of_frames(self.t_start, self.t_end, self.rate)
        duration = 1. / self.rate
        for w in win.windows[:-1]:
            self.assertAlmostEqual(w.duration(), duration)

    def test_against_old_concat_of_frames(self):
        win = concat_of_frames(self.t_start, self.t_end, self.rate)
        old_win = self.old_concat_of_frames()
        self.assertEqual(win.absolute_start, old_win.absolute_start)
        self.assertEqual(win.absolute_end, old_win.absolute_end)
        np.testing.assert_allclose([w.absolute_start for w in win.windows],
                                   [w.absolute_start for w in old_win.windows])
        np.testing.assert_allclose([w.absolute_end for w in win.windows],
                                   [w.absolute_end for w in old_win.windows])
