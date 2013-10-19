#!/usr/bin/env python2
# coding: utf-8


"""Main experiment presented in:
    **Learning semantic components from sub symbolic multi modal perception**
    O. Mangin and P.Y. Oudeyer
    *Joint IEEE International Conference on Development and Learning
     and on Epigenetic Robotics (ICDL EpiRob)*, Osaka (Japan) (2013)
"""


import sys

import numpy as np

from multimodal.db.choreo2 import load_features as load_motion
from multimodal.db.acorns import load_features as load_sound
from multimodal.lib.logger import Logger
from multimodal.pairing import associate_sound_motion
from multimodal.lib.metrics import kl_div, rev_kl_div, cosine_diff, frobenius
from multimodal.lib.utils import random_split
from multimodal.learner import MultimodalLearner
from multimodal.evaluation import (classify_NN, found_labels_to_score,
                                   chose_examples)


LOGGER = Logger()

PARAMS = {
    'acorns_speaker': 1,
    'motion_coef': 1.,  # Data normalization
    'sound_coef': .0008,  # Data normalization
    #'language_coef': 50,
    'iter_train': 50,
    'iter_test': 50,
    'k': 50,
    }
DEBUG = True
LOGGER.store_global('params', PARAMS)
LOGGER.store_global('debug', DEBUG)

if len(sys.argv) > 1:
    out_file = sys.argv[1]
    LOGGER.filename = out_file
else:
    out_file = None
    print('No output file')

# Load data
Xsound = load_sound(1, PARAMS['acorns_speaker'])
Xmotion = load_motion()
# Generate pairing
label_association, labels, assoc_idx = associate_sound_motion(
        PARAMS['acorns_speaker'])
LOGGER.store_global('label-pairing', label_association)
LOGGER.store_global('sample-pairing', assoc_idx)
N_LABELS = len(label_association[0])
# Align data
Y = Xsound[[i[0] for i in assoc_idx]]
X = Xmotion[[i[1] for i in assoc_idx]]

if DEBUG:  # To check for stupid errors
    print('WARNING: Debug mode active, using subset of the database')
    X = X[:200, :11]
    Y = Y[:200, :10]
    labels = labels[:200]
# Extract examples for evaluation
examples = chose_examples([l for l in labels])
others = [i for i in range(len(labels)) if i not in examples]
Xex = X[examples, :]
Yex = Y[examples, :]
X = X[others, :]
Y = Y[others, :]
ex_labels = [labels[i] for i in examples]
labels = [labels[i] for i in others]
n_samples = len(labels)
n_feat_motion = X.shape[1]
n_feat_sound = Y.shape[1]
# Also build label matrix
Zex = np.eye(N_LABELS)
Zex = Zex[ex_labels, :]

# Safety...
assert(ex_labels == range(N_LABELS))
assert(n_samples == X.shape[0])
assert(n_samples == Y.shape[0])
assert(all([l in range(10) for l in labels]))


for train, test in random_split(n_samples, .1):
#for train, test in [random_split(n_samples, .1).next()]:
#for train, test in leave_one_out(n_samples):
    LOGGER.new_experiment()
    # Extract train and test matrices
    Xtrain = X[train, :]
    Ytrain = Y[train, :]
    Xtest = X[test, :]
    Ytest = Y[test, :]
    test_labels = [labels[t] for t in test]

    # Init Learner
    learner = MultimodalLearner(
            ['motion', 'sound'],
            [n_feat_motion, n_feat_sound],
            [PARAMS['motion_coef'], PARAMS['sound_coef']],
            PARAMS['k']
            )
    # Train
    learner.train([Xtrain, Ytrain], PARAMS['iter_train'])
    # Get coefs
    coefs_motion = learner.reconstruct_internal(
            'motion', Xtest, PARAMS['iter_test'])
    coefs_motion_ex = learner.reconstruct_internal(
            'motion', Xex, PARAMS['iter_test'])
    coefs_sound = learner.reconstruct_internal(
            'sound', Ytest, PARAMS['iter_test'])
    coefs_sound_ex = learner.reconstruct_internal(
            'sound', Yex, PARAMS['iter_test'])

    # Get other modalities
    motion_as_sound = learner.reconstruct_modality('sound', coefs_motion)
    motion_as_sound_ex = learner.reconstruct_modality('sound', coefs_motion_ex)
    sound_as_motion = learner.reconstruct_modality('motion', coefs_sound)
    sound_as_motion_ex = learner.reconstruct_modality('motion', coefs_sound_ex)

    # Evaluate coefs
    for mod, coefs, coefs_ex in [
            ('motion', coefs_motion, coefs_motion_ex),
            ('motion2sound', coefs_motion, coefs_sound_ex),
            ('sound', coefs_sound, coefs_sound_ex),
            ('sound2motion', coefs_sound, coefs_motion_ex),
            ('motion2sound_sound', motion_as_sound, Yex),
            ('motion2sound_motion', Xtest, sound_as_motion_ex),
            ('sound2motion_motion', sound_as_motion, Xex),
            ('sound2motion_sound', Ytest, motion_as_sound_ex),
            ]:
        for metric, suffix in zip([kl_div, rev_kl_div, frobenius, cosine_diff],
                                  ['', '_bis', '_frob', '_cosine']):
            # Perform recognition
            found = classify_NN(coefs, coefs_ex, ex_labels, metric)
            # Conpute score
            LOGGER.store_result("score_%s%s" % (mod, suffix),
                                found_labels_to_score(test_labels, found))

#    for mod in ['motion', 'sound']:
#        # Evaluate various sparseness
#        LOGGER.store_result("sparseness_dico_%s" % mod,
#                            hoyer_sparseness(learner.get_dico(modality=mod)))
#        LOGGER.store_result("sparseness_coef_%s" % mod,
#                            hoyer_sparseness(coefs))

    print '.',
    sys.stdout.flush()
print


# Note Random_split creates subset of different sizes (actually only the last
# one might be smaller) which makes the average not fully meaningful.
LOGGER.print_all_results()


if out_file is not None:
    LOGGER.save()
