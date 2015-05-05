#!/usr/bin/env python
# coding: utf-8


"""Experiment for image and sound.
Additionaly the recognition of labels along the sentences is evaluated.
"""


import os
import sys
from shutil import rmtree
from tempfile import mkdtemp
from multiprocessing import Pool

import scipy.sparse as sp
from scipy.io import loadmat

from multimodal.experiment import TwoModalitiesExperiment
from multimodal.db.objects import ObjectsLoader
from multimodal.db.acorns import Year1Loader as AcornsLoader
from multimodal.lib.window import (concat_from_list_of_wavs, BasicTimeWindow,
                                   ConcatTimeWindow, slider)
from multimodal.learner import MultimodalLearner
from multimodal.lib.metrics import cosine_diff
from multimodal.evaluation import all_distances
from multimodal.features.hac import hac
from multimodal.local import CONFIG


CODEBOOK_PATH = os.path.join(CONFIG['feat-dir'], "vctk_codebook.mat")


DEBUG = False
if len(sys.argv) > 1 and sys.argv[1] == '--debug':
    DEBUG = True
    sys.argv.pop(1)

exp = TwoModalitiesExperiment({'objects': ObjectsLoader(['SURF', 'color']),
                               'sound': AcornsLoader(1)},
                              50, 50, 50, debug=DEBUG, run_mode='single')

if len(sys.argv) > 1:
    path = ''
    if len(sys.argv) > 2:
        path = os.path.expanduser(sys.argv[2])
    path = os.path.join(os.getcwd(), path)
    exp.set_out_path_and_name(path, sys.argv[1])

# Regular run
exp.run()
exp.print_result_table()

WIDTH = .5
SHIFT = .1

# Modality indexes
sound_modality = exp.modalities.index('sound')
objects_modality = exp.modalities.index('objects')

# Perform additional evaluation
test_idx = exp.logger.get_last_value('test')  # TODO this is unused, FIX
# WARNING: currently the code computes associations for train & test examples
test_labels = [exp.labels[t] for t in test_idx]
assoc_idx = exp.logger.get_value('sample_pairing')
if DEBUG:
    test_idx = test_idx[:10]
    test_labels = test_labels[:10]
    assoc_idx = assoc_idx[:10]
sound_loader = exp.loaders[sound_modality]

# Will compute matching windows for all wavs
test_wavs = [sound_loader.records[i[sound_modality]].get_audio_path()
             for i in assoc_idx]
print('Building time windows from wav files...')
test_sound_wins = concat_from_list_of_wavs(test_wavs)
# Also build index and label windows
test_sound_idx_wins = ConcatTimeWindow([
    BasicTimeWindow(w.absolute_start, w.absolute_end, obj=i[sound_modality])
    for i, w in zip(assoc_idx, test_sound_wins.windows)
    ])
test_sound_labels_wins = test_sound_idx_wins.copy()
for w, l in zip(test_sound_labels_wins.windows, test_labels):
    w.obj = l

# Sliding windows
sliding_wins = [test_sound_wins.get_subwindow(ts, te)
                for ts, te in slider(test_sound_wins.absolute_start,
                                     test_sound_wins.absolute_end,
                                     WIDTH, SHIFT)
                ]


codebooks = loadmat(CODEBOOK_PATH)['codebooks'].flatten().tolist()


def process_win(w):
    array_w = w.to_array_window()
    win_hac = hac(array_w.array, array_w.rate, codebooks)
    return sp.csc_matrix(win_hac)

# Create tmp dir
TMPDIR = mkdtemp()
pool = Pool()
print('Processing {} windows... (using {} processes).'.format(
    len(sliding_wins), pool._processes))
vectors = pool.map(process_win, sliding_wins)
print('Finished.')
# Cleanup
rmtree(TMPDIR)

# Compute features on windows...
X_test = sp.vstack(vectors)
if DEBUG:
    X_test = X_test.tocsr()[:, :exp.n_features[sound_modality]]

learner = MultimodalLearner(exp.modalities, exp.n_features,
                            exp.coefs, exp.k)
learner.dico = exp.logger.get_last_value('dictionary')

test_coefs_sound = learner.reconstruct_internal('sound', X_test, exp.iter_test)
ex_coefs_objects = learner.reconstruct_internal(
    'objects', exp.data_ex[objects_modality], exp.iter_test)

# Perform recognition
distances = all_distances(test_coefs_sound, ex_coefs_objects, cosine_diff)
exp.logger.store('sliding_distances', distances)
exp.logger.store('label_ex', exp.labels_ex)
try:
    exp.logger.save()
except exp.logger.NoFileError:
    print('Not saving logs: no destination was provided.')
