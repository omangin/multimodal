from textwrap import wrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

from .lib.plot import plot, boxplot, plot_var, legend
from .lib.window import (BasicTimeWindow, ConcatTimeWindow, TimeOutOfBound)


COLORS = {'motion': '#006EB8',
          'image': 'orange',
          'sound': '#3C8031',
          }

PAIRS_COLORS = {('image', 'motion'): '#006EB8',
                ('motion', 'image'): 'orange',
                ('image', 'sound'): '#3C8031',
                ('sound', 'image'): 'violet',
                ('motion', 'sound'): 'Red',
                ('sound', 'motion'): '#FBB982',
                }


def figure(**kwargs):
    if 'frameon' not in kwargs:
        kwargs['frameon'] = False
    return plt.figure(**kwargs)


def get_color_for_modalities(mods1, mods2, colors=None):
    if colors == 'pairs':
        return PAIRS_COLORS[mods1[0], mods2[0]]
    else:
        return COLORS[mods2[0]]


def mod_to_mod_str(mod1, mod2):
    return '{} $\\rightarrow$ {}'.format(mod1, mod2)


def all_pairs(iterable):
    return [([i], [j]) for i in iterable for j in iterable if i != j]


def combinations(iterable):
    pairs = all_pairs(iterable)
    two_to_one = []
    for i in iterable:
        two_to_one.append(([j for j in iterable if j != i], [i]))
    return pairs + two_to_one


def plot_one_curve(loggers, mods, ks, metric='', linestyle='-',
                   var_style='fill'):
    res = [loggers[i].get_values("score_{}2{}".format('_'.join(mods[0]),
                                                      '_'.join(mods[1])))
           for i, k in enumerate(ks)]
    return plot_var(res, x=ks, label=mod_to_mod_str(*mods),
                    color=get_color_for_modalities(*mods, colors='pairs'),
                    linewidth=2, marker='o', var_style=var_style,
                    linestyle=linestyle)


def plot_k_graphs(loggers, ks, title='', metric=''):
    figure()
    for mod1, mod2 in loggers:
        plot_one_curve(loggers[(mod1, mod2)], ([mod1], [mod2]), ks,
                       metric=metric)
        plot_one_curve(loggers[(mod1, mod2)], ([mod2], [mod1]), ks,
                       metric=metric)
    legend()
    plt.title(title)
    plt.show()


def plot_2k_graphs(loggers, ks, title='', metric=''):
    fig = figure()
    lines = []
    for (logs, linestyle) in zip(loggers, ['-', '--']):
        lines.append({})
        for mod1, mod2 in logs:
            l1 = plot_one_curve(logs[(mod1, mod2)], ([mod1], [mod2]), ks,
                                metric=metric, linestyle=linestyle,
                                var_style='bar')
            l2 = plot_one_curve(logs[(mod1, mod2)], ([mod2], [mod1]), ks,
                                metric=metric, linestyle=linestyle,
                                var_style='bar')
            lines[-1][mod_to_mod_str(mod1, mod2)] = l1
            lines[-1][mod_to_mod_str(mod2, mod1)] = l2
    legend1 = legend(lines[0].values(), lines[0].keys(), loc=4)
    legend([Line2D(range(5), range(5), color='black', linestyle='-'),
            Line2D(range(5), range(5), color='black', linestyle='--')],
           ['Trained on two modalities.', 'Trained on all three modalities.'])
    ax = plt.gca()
    ax.add_artist(legend1)
    ax.set_xlabel('k')
    ax.set_ylabel('Cross-modal association score')
    plt.title(title)
    return fig


def plot_boxes_one_exp(logger, mods_to_mods, colors=None, xticks=False):
    ax = plt.gca()
    ax.yaxis.grid(True, color='lightgrey')
    plt.ylim(0, 1)
    vals = [logger.get_values("score_{}2{}_cosine".format('_'.join(mods1),
                                                          '_'.join(mods2)))
            for mods1, mods2 in mods_to_mods]
    boxes = boxplot(vals, xticklabels=[])['boxes']
    polygons = []
    for box, mods in zip(boxes, mods_to_mods):
        coords = zip(box.get_xdata(), box.get_ydata())
        fcolor = get_color_for_modalities(*mods, colors=colors)
        p = plt.Polygon(coords, facecolor=fcolor, alpha=.8)
        ax.add_patch(p)
        polygons.append(p)
    labels = ['{} $\\rightarrow$ {}'.format(', '.join(mod1), ', '.join(mod2))
              for mod1, mod2 in mods_to_mods]
    if xticks:
        plt.xticks(range(1, 1 + len(mods_to_mods)),
                   labels, rotation=25, ha='right')
    return polygons, labels


def plot_boxes_one_exp_pairs(logger, mods, colors=None, xticks=False):
    return plot_boxes_one_exp(logger, all_pairs(mods), colors=colors,
                              xticks=xticks)


def plot_boxes(loggers2, logger3):
    fig = figure()
    nb_pairs = len(loggers2)
    for i, mods in enumerate(loggers2):
        if i == 0:
            ax1 = plt.subplot(1, 2 * nb_pairs, 1 + i)
            ax1.set_ylabel('Cross-modal association score')
        else:
            ax = plt.subplot(1, 2 * nb_pairs, 1 + i, sharey=ax1)
            ax.label_outer()
            ax.spines['left'].set_visible(False)
        log = loggers2[mods]
        plot_boxes_one_exp_pairs(log, mods, colors='pairs')
        plt.title(', '.join(mods))
    ax = plt.subplot(1, 2, 2, sharey=ax1)
    ax.label_outer()
    ax.spines['left'].set_visible(False)
    mods = logger3.get_value('modalities')
    polygons, labels = plot_boxes_one_exp_pairs(logger3, mods, colors='pairs')
    legend(polygons, labels, fig=fig,  ncol=3, loc='lower center',
           bbox_to_anchor=(.45, .05))
    plt.title(', '.join(mods))
    return fig


def plot_boxes_all_mods(logger3):
    fig = figure()
    mods = logger3.get_value('modalities')
    polygons, labels = plot_boxes_one_exp(logger3, combinations(mods))
    plt.gca().set_ylabel('Cross-modal association score')
    plt.title(', '.join(mods))
    legend(polygons, labels, ncol=3, loc='lower center')
    return fig


def plot_boxes_by_feats(loggers):
    fig = figure()
    mods = ('image', 'sound')
    nb_feats = len(loggers)
    for i, feats in enumerate(loggers):
        if i == 0:
            ax1 = plt.subplot(1, nb_feats, 1 + i)
        else:
            ax = plt.subplot(1, nb_feats, 1 + i, sharey=ax1)
            ax.label_outer()
            ax.spines['left'].set_visible(False)
        log = loggers[feats]
        polygons, labels = plot_boxes_one_exp_pairs(log, mods, colors='pairs')
        plt.title('\n'.join(wrap(', '.join(feats).replace('_', '-'), 15)),
                  fontsize='x-small')
    ax1.set_ylabel('Cross-modal association score')
    legend(polygons, labels, fig=fig,  ncol=3, loc='lower center',
           bbox_to_anchor=(.4, .045))
    return fig


# Plots for sliding windows

# TODO: use fact that windows are sorted to opimize filter
class ScorePlot(object):

    def __init__(self, record_wins, sliding_wins, similarities, example_labels,
                 draw_sentence_boundaries=True, use_relative_time=False):
        self.current = BasicTimeWindow(0., 20.)
        self.records = record_wins  # ConcatTimeWindow
        self.sliding = sliding_wins  # List
        self.similarity = [BasicTimeWindow(w.absolute_start, w.absolute_end,
                                           similarities[i, :])
                           for i, w in enumerate(self.sliding)]
        self.fig, self.main_ax = plt.subplots()
        self.example_labels = example_labels
        self.draw_sentence_boundaries = draw_sentence_boundaries
        self.use_relative_time = use_relative_time

    def is_test(self, record):
        """Returns whether sentence is from test set.

           Mainly used to print its transcription in bold.
           Redefine this function for another behavior.
        """
        return True

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

    def relative_time(self, time):
        """Return relative or absolute time depending on setup.
        """
        if self.use_relative_time:
            return time - self.current.absolute_start
        else:
            return time

    def relative(self, time, window):
        if time == 'start':
            t = window.absolute_start
        elif time == 'end':
            t = window.absolute_end
        elif time == 'mean':
            t = window.mean_time()
        else:
            raise ValueError()
        return self.relative_time(t)

    def draw(self):
        filtered_windows = self.filter_windows(self.sliding)
        times = [self.relative('mean', w) for w in filtered_windows]
        win_boundaries = [(max(self.relative('start', w),
                               self.relative('start', self.current)),
                           min(self.relative('end', w),
                               self.relative('end', self.current)))
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
                self.relative('mean', w), -.05, w.obj.trans,
                horizontalalignment='center',
                fontdict={'color': 'black' if self.is_test(w.obj) else 'gray'})
            if self.draw_sentence_boundaries:
                self.main_ax.axvline(x=self.relative('end', w),
                                     linewidth=2, linestyle='-', color='gray')
        legend(plots, self.example_labels, ax=self.main_ax)
        self.main_ax.set_xbound(self.relative('start', self.current),
                                self.relative('end', self.current))
        self.fig.canvas.draw_idle()


class InteractivePlot(object):

    def __init__(self, record_wins, sliding_wins, example_labels,
                 similarities):
        self.score_plot = ScorePlot(record_wins, sliding_wins, example_labels,
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


def plot_one_sentence(record_win, sliding_wins, example_labels, similarities):
    score_plot = ScorePlot(ConcatTimeWindow([record_win]), sliding_wins,
                           example_labels, similarities,
                           draw_sentence_boundaries=False,
                           use_relative_time=True)
    score_plot.current.absolute_start = record_win.absolute_start
    score_plot.current.absolute_end = record_win.absolute_end
    score_plot.draw()
