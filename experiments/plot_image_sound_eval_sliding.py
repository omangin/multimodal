import os

import numpy as np
import matplotlib as plt

from multimodal.lib.metrics import mutual_information
from multimodal.lib.logger import Logger
from multimodal.lib.window import (BasicTimeWindow, ConcatTimeWindow,
                                   concat_from_list_of_wavs,
                                   slider)
from multimodal.db.acorns import Year1Loader as AcornsLoader
from multimodal.plots import InteractivePlot, plot_one_sentence


WIDTH = .5
SHIFT = .1

WORKDIR = os.path.expanduser('~/work/data/results/quick/')

sound_loader = AcornsLoader(1)

logger = Logger.load(os.path.join(WORKDIR, 'sliding'))

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
        hist = [[h[i, j],
                 sum(h[i, :j]) + sum(h[i, (j + 1):])],
                [p_labels[j] - h[i, j],
                 sum(p_labels - h[i, :]) - p_labels[j] + h[i, j]],
                ]
        word_label_info[i, j] = mutual_information(np.array(hist))

plt.interactive(True)
plt.style.use('ggplot')
example_labels = [sound_labels[i] for i in logger.get_last_value('label_ex')]
myplot = InteractivePlot(record_wins, sliding_wins, similarities,
                         example_labels, is_test=lambda r: r in test_records)

# Prepare for plotting sentence results in files
DESTDIR = os.path.join(WORKDIR, 'sliding_win_plots')
if not os.path.exists(DESTDIR):
    os.mkdir(DESTDIR)
PLOT_PARAMS = {
    'font.family': 'serif',
    'font.size': 9.0,
    'font.serif': 'Computer Modern Roman',
    'text.usetex': 'True',
    'text.latex.unicode': 'True',
    'axes.titlesize': 'large',
    'axes.labelsize': 'large',
    'legend.fontsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'path.simplify': 'True',
    'savefig.bbox': 'tight',
    'figure.figsize': (8, 4),
}
SENTENCE_PLOT_RC = {
    'window_boundaries_color': 'gray',
    'window_boundaries_line_width': 1,
}
with plt.rc_context(rc=PLOT_PARAMS):
    plt.pyplot.interactive(False)
    # Plot sentence results to disk
    test_record_wins = [w for w in record_wins.windows
                        if w.obj in test_records]
    for r in test_record_wins[:5]:
        path = os.path.join(DESTDIR, '{}.svg'.format(
            r.obj.audio.split('.')[0]))
        score_plot = plot_one_sentence(r, sliding_wins, similarities,
                                       example_labels,
                                       plot_rc=SENTENCE_PLOT_RC)
        score_plot.fig.savefig(path, transparent=True)
        print('Written: {}.'.format(path))

#plt.pyplot.figure()
##most_info = np.nonzero(np.max(word_label_info, axis=1) > .04)[0]
##p = pcolormesh(word_label_info[most_info, :],
##               xticklabels=all_labels,
##               yticklabels=[all_words[i] for i in most_info])
#p = pcolormesh(word_label_info, xticklabels=all_labels, yticklabels=all_words)
#plt.pyplot.colorbar(p)
#
##plot_one_sentence(record_wins[0], sliding_wins, sound_labels, similarities)
