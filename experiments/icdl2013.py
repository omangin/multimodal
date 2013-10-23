#!/usr/bin/env python2
# coding: utf-8


"""Main experiment presented in:
    **Learning semantic components from sub symbolic multi modal perception**
    O. Mangin and P.Y. Oudeyer
    *Joint IEEE International Conference on Development and Learning
     and on Epigenetic Robotics (ICDL EpiRob)*, Osaka (Japan) (2013)
"""


import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))

from multimodal.db.choreo2 import load_features as load_motion
from multimodal.db.acorns import load_features as load_sound
from multimodal.lib.logger import Logger
from multimodal.lib.metrics import kl_div, rev_kl_div, cosine_diff, frobenius
from multimodal.lib.utils import random_split
from multimodal.pairing import associate_sound_motion
from multimodal.learner import MultimodalLearner
from multimodal.evaluation import (classify_NN, found_labels_to_score,
                                   chose_examples)


LOGGER = Logger()

PARAMS = {
    'acorns_speaker': 0,
    'motion_coef': 1.,  # Data normalization
    'sound_coef': .0008,  # Data normalization
    #'language_coef': 50,
    'iter_train': 50,
    'iter_test': 50,
    'k': 50,
    }
LOGGER.store_global('params', PARAMS)

DEBUG = True
if len(sys.argv) > 1 and sys.argv[1] == '--debug':
    DEBUG = True
    sys.argv.pop(1)
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
Xs = {'sound': Xsound[[i[0] for i in assoc_idx]],
      'motion': Xmotion[[i[1] for i in assoc_idx]]}
MODALITIES = Xs.keys()

if DEBUG:  # To check for stupid errors
    print('WARNING: Debug mode active, using subset of the database')
    for mod in Xs:
        Xs[mod] = Xs[mod][:200, :11]
    labels = labels[:200]
# Extract examples for evaluation
examples = chose_examples([l for l in labels])
others = [i for i in range(len(labels)) if i not in examples]
Xex = {mod: Xs[mod][examples, :] for mod in MODALITIES}
X = {mod: Xs[mod][others, :] for mod in MODALITIES}
n_feats = {mod: X[mod].shape[1] for mod in MODALITIES}
ex_labels = [labels[i] for i in examples]
labels = [labels[i] for i in others]
n_samples = len(labels)

# Safety...
if not DEBUG:
    assert(ex_labels == range(N_LABELS))
for mod in MODALITIES:
    assert(n_samples == X[mod].shape[0])
assert(all([l in range(10) for l in labels]))


for train, test in random_split(n_samples, .1):
#for train, test in [random_split(n_samples, .1).next()]:  # for a single run
#for train, test in leave_one_out(n_samples):
    LOGGER.new_experiment()
    # Extract train and test matrices
    Xtrain = {mod: X[mod][train, :] for mod in MODALITIES}
    Xtest = {mod: X[mod][test, :] for mod in MODALITIES}
    test_labels = [labels[t] for t in test]

    # Init Learner
    learner = MultimodalLearner(
            MODALITIES,
            [n_feats[mod] for mod in MODALITIES],
            [PARAMS["%s_coef" % mod] for mod in MODALITIES],
            PARAMS['k'])
    # Train
    learner.train([Xtrain[mod] for mod in MODALITIES], PARAMS['iter_train'])

    # Get internal coefs
    internal_test = {mod: learner.reconstruct_internal(
        mod, Xtest[mod], PARAMS['iter_test']) for mod in MODALITIES}
    internal_ex = {mod: learner.reconstruct_internal(
                mod, Xex[mod], PARAMS['iter_test']) for mod in MODALITIES}

    # Reconstruct one modality from an other
    Xreco_test = {}
    Xreco_ex = {}
    for (m1, m2) in [('motion', 'sound'), ('sound', 'motion')]:
        key = "%s_as_%s" % (m1, m2)
        Xreco_test[key] = learner.reconstruct_modality(m2, internal_test[m1])
        Xreco_ex[key] = learner.reconstruct_modality(m2, internal_ex[m1])

    # Evaluate coefs
    to_test = []
    for (mod1, mod2) in [('motion', 'sound'), ('sound', 'motion')]:
        # Evaluate recognition in single modality
        to_test.append((mod1, internal_test[mod1], internal_ex[mod1]))
        # Evaluate one modality against the other:
        # - on internal coefficients
        to_test.append((mod1 + '2' + mod2, internal_test[mod1],
                        internal_ex[mod2]))
        # - original data for test mod1
        #   compared to reconstructed mod1 from mod2 reference examples
        to_test.append(("{}2{}_{}".format(mod1, mod2, mod1),
                        Xtest[mod1], Xreco_ex[mod2 + '_as_' + mod1]))
        # - reconstructed mod2 from mod1 test examples
        #   compared to original for mod2 reference examples
        to_test.append(("{}2{}_{}".format(mod1, mod2, mod2),
                        Xreco_test[mod1 + '_as_' + mod2], Xex[mod2]))

    for mod, coefs, coefs_ex in to_test:
        for metric, suffix in zip([kl_div, rev_kl_div, frobenius, cosine_diff],
                                  ['', '_bis', '_frob', '_cosine']):
            # Perform recognition
            found = classify_NN(coefs, coefs_ex, ex_labels, metric)
            # Conpute score
            LOGGER.store_result("score_%s%s" % (mod, suffix),
                                found_labels_to_score(test_labels, found))


def get_key(mod1, mod2, mod_comp, metric_key):
    return "score_{}2{}{}{}".format(
            mod1, mod2,
            '' if mod_comp == 'internal' else '_' + mod_comp,
            metric_key)


# Print result table
WIDTH = 13
table = " ".join(["{}" for _dummy in range(6)])

print('-' * (WIDTH * 6 + 5))
print('Modalities'.center(WIDTH * 3 + 2)
      + 'Score: avg (std)'.center(WIDTH * 3 + 2))
print('-' * (WIDTH * 3 + 2) + ' ' + '-' * (WIDTH * 3 + 2))
print table.format(*[s.center(WIDTH)
                     for s in ['Test', 'Reference', 'Comparison',
                               'KL', 'Euclidean', 'Cosine']])
print(' '.join(['-' * WIDTH] * 6))
for mods in [('sound', 'motion'), ('motion', 'sound')]:
    for mod_comp in ['internal', 'motion', 'sound']:
        mod_str = [mods[0].center(WIDTH), mods[1].center(WIDTH),
                   mod_comp.center(WIDTH)]
        res_str = [("%.3f (%.3f)" % LOGGER.get_stats(get_key(mods[0], mods[1],
                                                     mod_comp, metr)))
                   for metr in ['', '_frob', '_cosine']]
        print table.format(*[s.center(WIDTH) for s in (mod_str + res_str)])
print('-' * (WIDTH * 6 + 5))


if out_file is not None:
    LOGGER.save()
