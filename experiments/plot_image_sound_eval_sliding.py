import os

import numpy as np

from multimodal.lib.metrics import mutual_information
from multimodal.lib.logger import Logger
from multimodal.lib.plot import plot, pcolormesh, legend
from multimodal.lib.window import (BasicTimeWindow, ConcatTimeWindow,
                                   TimeOutOfBound, concat_from_list_of_wavs,
                                   slider)
from multimodal.db.acorns import Year1Loader as AcornsLoader


WIDTH = .5
SHIFT = .1


sound_loader = AcornsLoader(1)

logger = Logger.load(os.path.expanduser('~/work/data/results/quick/sliding'))

# Build windows to access time and metadata
sound_modality = logger.get_value('modalities').index('sound')
assoc_idx = logger.get_value('sample_pairing')
test_idx = [assoc_idx[i][sound_modality]
            for i in logger.get_last_value('test')]
test_records = set([sound_loader.records[i] for i in test_idx])
test_wavs = [sound_loader.records[i[sound_modality]].get_audio_path()
             for i in assoc_idx]
print('Building time windows from wav files...')
test_sound_wins = concat_from_list_of_wavs(test_wavs)
# Also build index windows
record_wins = ConcatTimeWindow([
    BasicTimeWindow(w.absolute_start, w.absolute_end,
                    obj=sound_loader.records[i[sound_modality]])
    for i, w in zip(assoc_idx, test_sound_wins.windows)
    ])
# Sliding windows
sliding_wins = [test_sound_wins.get_subwindow(ts, te)
                for ts, te in slider(test_sound_wins.absolute_start,
                                     test_sound_wins.absolute_end,
                                     WIDTH, SHIFT)
                ]


similarities = -logger.get_last_value('sliding_distances')
sound_labels = logger.get_value('label_pairing')[sound_modality]

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt


# TODO: use fact that windows are sorted to opimize filter
class ScorePlot(object):

    draw_sentence_boundaries = True

    def __init__(self, record_wins, sliding_wins, sound_labels, similarities):
        self.current = BasicTimeWindow(0., 20.)
        self.records = record_wins  # ConcatTimeWindow
        self.sliding = sliding_wins  # List
        self.sound_labels = sound_labels
        self.similarity = [BasicTimeWindow(w.absolute_start, w.absolute_end,
                                           similarities[i, :])
                           for i, w in enumerate(self.sliding)]
        self.fig, self.main_ax = plt.subplots()

    def subwindows(self, win):
        return win.get_subwindow(self.current.absolute_start,
                                 self.current.absolute_end).windows

    def time_in_current(self, time):
        try:
            self.current.check_time(time)
            return True
        except TimeOutOfBound:
            return False

    def filter_windows(self, windows):
        # Not optimal knowing that windows are sorted...
        return [w for w in windows if self.time_in_current(w.mean_time())]

    def draw(self):
        filtered_windows = self.filter_windows(self.sliding)
        times = [w.mean_time() for w in filtered_windows]
        win_boundaries = [(max(w.absolute_start, self.current.absolute_start),
                           min(w.absolute_end, self.current.absolute_end))
                          for w in filtered_windows]
        similarities = np.array(
                [w.obj for w in self.filter_windows(self.similarity)])
        # Clean and Plot
        self.main_ax.cla()
        # Plot window boundaries
        for (t, s) in zip(win_boundaries, similarities):
            y = max(s)
            self.main_ax.plot(t, (y, y), color='white', linewidth=2)
        # Plot scores
        plots = plot(times, similarities,
                     linestyle='-', marker='o', ax=self.main_ax)
        # Plot sentence text and boundaries
        for w in self.subwindows(self.records):
            self.main_ax.text(
                w.mean_time(), -.05, w.obj.trans,
                horizontalalignment='center',
                fontdict={'color': 'black' if w.obj in test_records
                          else 'gray'})
            if self.draw_sentence_boundaries:
                self.main_ax.axvline(x=w.absolute_end, linewidth=2,
                                     linestyle='-', color='gray')
        legend(plots,
               [sound_labels[i] for i in logger.get_last_value('label_ex')],
               ax=self.main_ax)
        self.main_ax.set_xbound(self.current.absolute_start,
                                self.current.absolute_end)
        self.fig.canvas.draw_idle()


class InteractivePlot(object):

    def __init__(self, record_wins, sliding_wins, sound_labels, similarities):
        self.score_plot = ScorePlot(record_wins, sliding_wins, sound_labels,
                                    similarities)
        self.fig, self.main_ax = self.score_plot.fig, self.score_plot.main_ax
        plt.subplots_adjust(bottom=0.2)
        self.score_plot.draw()
        max_time = self.score_plot.records.absolute_end
        ax_s = plt.axes([0.15, 0.1, 0.75, 0.03])
        ax_w = plt.axes([0.15, 0.05, 0.75, 0.03])
        self.slider_start = Slider(ax_s, 'Start', 0., max_time, valinit=0.)
        self.slider_start.on_changed(self.update)
        self.slider_width = Slider(ax_w, 'Width', 0., 30., valinit=15.)
        self.slider_width.on_changed(self.update)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def update(self, val):
        self.score_plot.current.absolute_start = self.slider_start.val
        width = self.slider_width.val
        self.score_plot.current.absolute_end = (
                self.score_plot.current.absolute_start + width)
        self.score_plot.draw()

    def on_key(self, event):
        if event.key == 'right':
            direction = 1
        elif event.key == 'left':
            direction = -1
        else:
            return
        self.slider_start.set_val(self.slider_start.val
                                  + direction * .5 * self.slider_width.val)
        self.update(None)


def plot_one_sentence(record_win, sliding_wins, sound_labels, similarities):
    score_plot = ScorePlot(ConcatTimeWindow([record_win]), sliding_wins,
                           sound_labels, similarities)
    score_plot.draw_sentence_boundaries = False
    score_plot.current.absolute_start = record_win.absolute_start
    score_plot.current.absolute_end = record_win.absolute_end
    score_plot.draw()


def word_histo_by_label(records, labels):
    assert(len(records) == len(labels))
    labels = [w.lower() for w in labels]
    all_labels = list(set(labels))
    words_by_record = [r.trans.strip('.?!').lower().split() for r in records]
    all_words = set(sum(words_by_record, []))
    # Re-order with all labels first
    all_words = all_labels + sorted(list(all_words.difference(all_labels)))
    word_idx = [[all_words.index(w) for w in set(t)] for t in words_by_record]
    word_counts = [np.bincount(t, minlength=len(all_words)) for t in word_idx]
    counts_by_labels = [[] for _ in all_labels]
    for l, h in zip(labels, word_counts):
        counts_by_labels[all_labels.index(l)].append(h)
    h = np.vstack([np.sum(1. * np.vstack(l), axis=0)
                   for l in counts_by_labels]).T
    h /= len(records)
    p_labels = 1. * np.array([len(l) for l in counts_by_labels]) / len(records)
    return h, p_labels, all_labels, all_words


h, p_labels, all_labels, all_words = word_histo_by_label(
    sound_loader.records, sound_loader.get_labels())
word_label_info = np.zeros(h.shape)
for i in range(len(all_words)):
    for j in range(len(all_labels)):
        hist = [[h[i, j], sum(h[i, :j]) + sum(h[i, (j + 1):])],
                [p_labels[j] - h[i, j], sum(p_labels - h[i, :]) - p_labels[j] + h[i, j]]]
        word_label_info[i, j] = mutual_information(np.array(hist))

plt.interactive(True)
plt.style.use('ggplot')
myplot = InteractivePlot(record_wins, sliding_wins, sound_labels, similarities)

test_record_wins = [w for w in record_wins.windows if w.obj in test_records]
for r in test_record_wins[:10]:
    plot_one_sentence(r, sliding_wins, sound_labels, similarities)

plt.figure()
#most_info = np.nonzero(np.max(word_label_info, axis=1) > .04)[0]
#p = pcolormesh(word_label_info[most_info, :],
#               xticklabels=all_labels,
#               yticklabels=[all_words[i] for i in most_info])
p = pcolormesh(word_label_info, xticklabels=all_labels, yticklabels=all_words)
plt.colorbar(p)

#plot_one_sentence(record_wins[0], sliding_wins, sound_labels, similarities)
