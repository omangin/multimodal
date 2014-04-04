import bisect

import numpy as np
from scipy.io import wavfile
from math import ceil, floor


TOL = 1.e-8
TOL_INT = 1.e-3
# Note:
# The first tolerance value is used to compare time values,
# the second for integers.
# Typically the integers are indexes in an array of samples and thus the two
# tolerance values have different scale such that:
# TOL * SAMPLE_RATE \sim TOL_INT.
# The dependancy to sample rate is made explicit in SampledTimeWindow and
# _arange.


def wavread(filepath):
    """Open wave file. See scikits.io.wavfile.read.
    """
    with open(filepath, 'r') as f:
        sr, data = wavfile.read(f)
    return sr, data


def _to_approx_int(x, tol=TOL_INT, above=False):
    f = int(floor(x))
    c = int(ceil(x))
    if c - x < tol:
        return c
    elif x - f < tol:
        return f
    elif above:
        return c
    else:
        return f


def _snap_above(x, ref, tol=TOL):
    return ref if ref - tol < x else x


def _arange(left, right, step, include, tol=TOL):
    """Consistent arange.
       Right border is included iif approximately step | right - left.
    """
    n_steps = _to_approx_int((right - left) * 1. / step, tol=(tol / step),
                             above=False)
    if n_steps < 0:
        return []
    else:
        l = [left + i * step for i in range(n_steps)]
        last = _snap_above(left + n_steps * step, right, tol=tol)
        if include or last != right:
            l.append(last)
        return l


class TimeOutOfBound(Exception):
    pass


class TimeWindow(object):

    """Time window.
    """

    def __repr__(self):
        return "[{}->{}]".format(self.absolute_start, self.absolute_end)

    def duration(self):
        return self.absolute_end - self.absolute_start

    def get_subwindow(self, t_start, t_end):
        """Returns the largest possible window included in given time range.
        """
        raise NotImplemented

    def relative_to_absolute_time(self, t):
        return self.absolute_start + t

    def absolute_to_relative_time(self, t):
        return t - self.absolute_start

    def move(self, delta_t):
        self.absolute_start += delta_t
        self.absolute_end += delta_t

    def check_time(self, t):
        if t < self.absolute_start:
            raise TimeOutOfBound("%f is before starting time %f"
                                 % (t, self.absolute_start))
        if t > self.absolute_end:
            raise TimeOutOfBound("%f is after end time %f"
                                 % (t, self.absolute_end))

    def copy(self):
        """Deep copy of the window objects."""
        raise NotImplemented


class BasicTimeWindow(TimeWindow):
    """Associates a time window to a basic object.
    """

    def __init__(self, absolute_start, absolute_end, obj=None):
        self.absolute_start = absolute_start
        self.absolute_end = absolute_end
        self.obj = obj

    def __repr__(self):
        return "[{}->{}: {}]".format(self.absolute_start, self.absolute_end,
                                     repr(self.obj))

    def get_subwindow(self, t_start, t_end):
        self.check_time(t_start)
        self.check_time(t_end)
        new = self.copy()
        new.absolute_start = t_start
        new.absolute_end = t_end if t_start < t_end else t_start
        return new

    def copy(self):
        return self.__class__(self.absolute_start, self.absolute_end,
                              obj=self.obj)


class SampledTimeWindow(TimeWindow):
    """Time window for discrete data (sequence of samples) at a fixed
    sample rate.

    The convention is that first sample is located at: start + .5 / rate.
    """

    def __init__(self, absolute_start, n_samples, rate):
        self.absolute_start = absolute_start
        self.rate = rate
        self.absolute_end = self.relative_to_absolute_time(
            self._samples_to_duration(n_samples))

    def __repr__(self):
        return "[{}->{} at {}]".format(self.absolute_start, self.absolute_end,
                                       self.rate)

    @property
    def n_samples(self):
        raise NotImplemented

    def _samples_to_duration(self, n_samples=1):
        return n_samples * 1. / self.rate

    def _get_index_after(self, t):
        self.check_time(t)
        return _to_approx_int(self.absolute_to_relative_time(t) * self.rate,
                              above=True, tol=TOL * self.rate)

    def _get_index_before(self, t):
        self.check_time(t)
        return _to_approx_int(self.absolute_to_relative_time(t) * self.rate,
                              tol=TOL * self.rate)

    def get_times(self):
        rel_times = (.5 + np.arange(self.n_samples)
                     ) * self._samples_to_duration()
        return self.absolute_start + rel_times


class ArrayTimeWindow(SampledTimeWindow):
    """Time window for array data. Each element from the array has a given
       length equal to 1 / self.rate.
    """

    def __init__(self, array, absolute_start, rate):
        SampledTimeWindow.__init__(self, absolute_start, len(array), rate)
        self.array = array

    @property
    def n_samples(self):
        return len(self.array)

    def get_subwindow(self, t_start, t_end):
        i_start = self._get_index_after(t_start)
        i_end = self._get_index_before(t_end)
        new_start = self.relative_to_absolute_time(
            self._samples_to_duration(i_start))
        return self.__class__(self.array[i_start:i_end], new_start, self.rate)

    def to_array_window(self):
        return self.copy()

    def copy(self):
        return self.__class__(self.array, self.absolute_start, self.rate)

    def write_as_wav(self, path):
        wavfile.write(path, self.rate, self.array)

    @staticmethod
    def concatenate(wins):
        """Note: does no check ordering of start times.
        """
        if not all([w.rate == wins[0].rate for w in wins]):
            raise ValueError('All windows must have the same rate.')
        arr = np.hstack([w.array for w in wins])
        return ArrayTimeWindow(arr, wins[0].absolute_start, wins[0].rate)


class WavFileTimeWindow(SampledTimeWindow):
    """Implements time window which data is stored in wav file.

    The time window may start after the beginning of the file and stop
    before its end. The file data is not kept loaded when not necessary.
    """

    def __init__(self, path_to_file, absolute_start, n_samples_and_rate=None):
        """n_samples_and_rate: (total nb of samples in file, sample rate)"""
        self.path_to_file = path_to_file
        if n_samples_and_rate is None:
            rate, samples = wavread(path_to_file)
            n_samples = len(samples)
        else:
            n_samples, rate = n_samples_and_rate
        SampledTimeWindow.__init__(self, absolute_start, n_samples, rate)
        self._start_after = 0  # start after n samples
        self._stop_index = n_samples  # stop after n samples

    @property
    def n_samples(self):
        return self._stop_index - self._start_after

    def _delay_start(self, new_t_start):
        """Delay start point in file so that starts at given absolute time.
        """
        shift = self._get_index_after(new_t_start)
        self.absolute_start += self._samples_to_duration(shift)
        self._start_after += shift

    def _stop_at(self, new_t_end):
        stop_after = self._get_index_before(new_t_end)
        self._stop_index = self._start_after + stop_after
        self.absolute_end = self.relative_to_absolute_time(
            self._samples_to_duration(stop_after))

    def _crop(self, t_start, t_end):
        self._delay_start(t_start)
        self._stop_at(t_end)

    def get_subwindow(self, t_start, t_end):
        subwin = self.copy()
        subwin._crop(t_start, t_end)
        return subwin

    def copy(self):
        new = self.__class__(
            self.path_to_file, self.absolute_start,
            n_samples_and_rate=(self._stop_index - self._start_after,
                                self.rate))
        new._start_after = self._start_after
        new._stop_index = self._stop_index
        return new

    def to_array_window(self):
        sr, samples = wavread(self.path_to_file)
        return ArrayTimeWindow(samples[self._start_after:self._stop_index],
                               self.absolute_start, sr)


class ConcatTimeWindow(TimeWindow):
    """Sequence of consecutive time windows."""

    def __init__(self, windows):
        if len(windows) < 1:
            raise ValueError('At least one window should be given.')
        self.windows = windows
        # Check for alignment (i.e. no gap or overlap between windows)
        starts = [w.absolute_start for w in self.windows[1:]]
        ends = self._ends()[:-1]
        if max([abs(a - b) for a, b in zip(ends, starts)] + [0]) > TOL:
            raise ValueError('Windows are not consecutive. '
                             'Consider using ConcatTimeWindow.align.')

    @property
    def absolute_start(self):
        return self.windows[0].absolute_start

    @property
    def absolute_end(self):
        return self.windows[-1].absolute_end

    def move(self, delta_t):
        for w in self.windows:
            w.move(delta_t)

    def get_subwindow(self, t_start, t_end):
        i_start = self._get_file_index_from_time(t_start)
        i_end = self._get_file_index_from_time(t_end)
        new_windows = [w.copy() for w in self.windows[i_start:1 + i_end]]
        new_windows[0] = new_windows[0].get_subwindow(
            t_start, new_windows[0].absolute_end)
        new_windows[-1] = new_windows[-1].get_subwindow(
            new_windows[-1].absolute_start, t_end)
        return self.__class__(new_windows)

    def to_array_window(self):
        """Only when windows have a to_array_method."""
        array_windows = [win.to_array_window() for win in self.windows]
        return ArrayTimeWindow.concatenate(array_windows)

    def copy(self):
        return self.__class__([win.copy() for win in self.windows])

    def get_subwindow_at(self, t):
        return self.windows[self._get_file_index_from_time(t)]

    def _ends(self):
        return [win.absolute_end for win in self.windows]

    def _get_file_index_from_time(self, t):
        self.check_time(t)
        return bisect.bisect_left(self._ends(), t)

    @classmethod
    def align(cls, windows, start=None):
        """Align start windows so that they cover consecutive time intervals.
        Changes start times but not durations and window order.
        """
        if start is None:
            start = windows[0].absolute_start
        t = start
        for w in windows:
            w.move(t - w.absolute_start)
            t += w.duration()
        return windows


def concat_from_list_of_wavs(files, start=None):
    file_windows = ConcatTimeWindow.align(
        [WavFileTimeWindow(f, 0.) for f in files], start=start)
    return ConcatTimeWindow(file_windows)


def slider(t_start, t_end, width, shift, partial=False, tol=TOL):
    """Iterator over start/end time of time windows covering
    [t_start, t_end].

    partial: bool (False)
        If set returns partial window (i.e. of width < param),
        else stop at last full window.
    """
    end_start = t_end if partial else t_end - width
    # if partial: do not include t_end
    # else: include t_end - width
    starts = _arange(t_start, end_start, shift, not partial, tol=tol)
    return [(t, _snap_above(t + width, t_end, tol=tol)) for t in starts]


def concat_of_frames(t_start, t_end, rate):
    """Produces a list of frames that cover the given time interval.
    The frame are generated at a given rate.
    Thus they all have the same duration, except for the last one.
    """
    duration = 1. / rate
    times = slider(t_start, t_end, duration, duration, partial=True)
    return ConcatTimeWindow([BasicTimeWindow(ts, te) for (ts, te) in times])
