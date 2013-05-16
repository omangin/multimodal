#!/usr/bin/env python2
# encoding: utf-8


"""Export paired feature matrices for Acorns and Choreo v2.

    Convention for exported data matrices: shape -> (nb_examples, nb_features)
"""


import os

import numpy as np
import scipy.sparse as sp
from scipy.cluster.vq import kmeans2, whiten
from scipy.io import savemat

from exp.db.acorns import load_features_and_labels as load_sound
from exp.db.choreo2 import load as load_motion_db
from exp.lib.tf_to_angles import angles_indices, db_to_VQ_hist_matrix


DATA_DIR = '/home/omangin/work/data/'
# Motion DB
MOTION_DB_FILE = DATA_DIR + 'capture/kinect2/primitive.json'
MOTION_FEATS = DATA_DIR + 'capture/features_primitives.npz'
# Sound DB
BLACKLIST = [[277]]  # Bad records (empty, etc.)
ACORNS_DIR = DATA_DIR + 'db/ACORNS/'
DB_DESCR = ACORNS_DIR + 'Acorns_Y1.xml'
HAC_FILE = ACORNS_DIR + 'HAC_acorns_Y1_S1' + '.mat'

OUT_DIR = DATA_DIR + 'db/'
OUT_EXT = '.mat'
PAIRED_OUTFILE = 'paired_features'
SPEAKER = 0


NB_BINS = 9

#KMEANS = 15
KMEANS = None

EPSILON = 1.e-8


## Load sound DB and features

Xsound, sound_labels, sound_label_names = load_sound(1, SPEAKER)
# Compute sound data matrix
print 'Sound features loaded.'
N_sound, N_feat_sound = Xsound.shape

# Re-map sound labels (for some speakers label 10 replaces label 5)
sound_label_set = list(set([l[0] for l in sound_labels]))
n_labels = len(sound_label_set)
old_to_new_sound_labels = [-1] * (1 + max(sound_label_set))
for i, l in enumerate(sound_label_set):
    old_to_new_sound_labels[l] = i
sound_labels = [old_to_new_sound_labels[l[0]] for l in sound_labels]
sound_label_names = [sound_label_names[l] for l in sound_label_set]
sound_label_list = range(len(sound_label_set))
print 'Sound labels:', sound_label_names
print


## Load motion DB and features

motion_db = load_motion_db()
ANGLE_INDICES = angles_indices(motion_db.marker_names)
if os.path.exists(MOTION_FEATS):
    # Load existing file of pre-computed features
    Xmotion = np.load(MOTION_FEATS)['Xmotion']
    assert (Xmotion.shape ==
            (len(motion_db.records), 3 * NB_BINS * len(ANGLE_INDICES)))
    print 'Motion features loaded.\n'
else:
    # Or generate them
    print 'Computing histograms...'
    # k=16 seems too big
    Xmotion = db_to_VQ_hist_matrix(motion_db, vel_delay=10, nb_bins=NB_BINS)
    #Xmotion = db_to_binned_hist_matrix(motion_db,
    #        vel_delay=10, nb_bins=NB_BINS)
    # TODO investigate that !!!! (19/03/2013)
    #print (Xmotion * (Xmotion < 0)).sum()
    #Xmotion *= (Xmotion > 0)
    np.savez(MOTION_FEATS, Xmotion=Xmotion)
    print "Motion features generated and save to: %s.\n" % MOTION_FEATS

assert np.all(Xmotion >= 0)

motion_label_set = motion_db.get_occuring_labels()
motion_labels = [r[1][0] for r in motion_db.records]
N_motion, N_feat_motion = Xmotion.shape


#  Check that labels are compatible
assert motion_label_set == set(sound_label_list)

#SOUND2MOTION_LABELS = list(MOTION_LABELS)
#MOTION2SOUND_LABELS = dict([(l, i)
#                             for i, l in enumerate(SOUND2MOTION_LABELS)])
#motion_labels_unified = [MOTION2SOUND_LABELS[l] for l in MOTION_LABELS]

print "Pairing sound and motion examples"
print("Total %d sound examples, %d motion examples, keeping %d"
        % (N_sound, N_motion, min(N_sound, N_motion)))
print("(%d sound features, %d motion features)"
        % (N_feat_sound, N_feat_motion))

## Pair sound and motion examples

# Organize labels by tag for pairing
sounds_by_tag = [[] for _ in xrange(10)]
motions_by_tag = [[] for _ in xrange(10)]
for (i, l) in enumerate(sound_labels):
    sounds_by_tag[l].append(i)
for (i, l) in enumerate(motion_labels):
    motions_by_tag[l].append(i)
print [len(l) for l in sounds_by_tag]
print [len(l) for l in motions_by_tag]
# Shuffle
for by_tag in [sounds_by_tag, motions_by_tag]:
    for l in by_tag:
        np.random.shuffle(l)

# Build pairs
pairs = []
nb_by_tags = []
for s, m in zip(sounds_by_tag, motions_by_tag):
    n = min(len(s), len(m))
    nb_by_tags.append(n)
    pairs.extend(zip(s[:n], m[:n]))
print "Number of pairs by tag: %s." % nb_by_tags

paired_Xsound = Xsound[[si for si, _ in pairs], :]
paired_Xmotion = Xmotion[[mi for _, mi in pairs], :]

## Build labels matrix

# just to be sure...
assert all([sound_labels[si] == motion_labels[mi] for si, mi in pairs])
# Build matrix
labels = [sound_labels[si] for si, _ in pairs]
Y = np.eye(n_labels)
# Cut unused last label
Y = Y[labels, :]
Y = sp.csr_matrix(Y)  # shape: (nb_ex, nb_labels)


## Shuffle data (since it is ordered by labels)

order = range(len(labels))
np.random.shuffle(order)
paired_Xsound = paired_Xsound[order, :]
paired_Xmotion = paired_Xmotion[order, :]
Y = Y[order, :]
labels = [labels[i] for i in order]

## Debug plots

#import matplotlib.pyplot as plt
#from pyUtils.plots import get_n_colors
#from mpl_toolkits.mplot3d import Axes3D
#
#def plot_motion_histo(histo, nb_dof, nb_bins):
#    dim, = histo.shape
#    nb_angles = 3 * nb_dof
#    assert nb_bins == dim / nb_angles
#    sqrt_nb_bins = np.sqrt(nb_bins)
#    if sqrt_nb_bins % 1 != 0:
#        raise ValueError('Wrong number of bins.')
#    rhistos = histo.reshape((3 * nb_dof, sqrt_nb_bins, sqrt_nb_bins))
#    img = np.hstack([rhistos[i, :, :] for i in range(3*nb_dof)])
#    plt.imshow(img)
#
#plt.figure()
#
#for j in range(30):
#    i = 5 * j
#    plt.subplot(15, 2, j + 1)
#    plot_motion_histo(Xmotion[:, i], len(ANGLE_INDICES), NB_BINS)
#    l = labels[i][0]
#    plt.title("Label %d: %s" % (l, motion_db.label_descriptions[l]))
#plt.show()
#
#
## Plot PCA of data
#zou = Xmotion - np.mean(Xmotion, axis=0)[np.newaxis, :]
#vals, vects = np.linalg.eig(np.cov(zou))
#zde = np.real(np.dot(vects.T, zou))
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#colors = get_n_colors(10)
#ax.scatter(zde.T[:, 0], zde.T[:, 1], zde.T[:, 2],
#           c=[colors[l[0]] for l in labels])
#plt.show()


# Hack to get reasonable values
MIN_DIST = 50.
# Eventually transform data by k-means
if KMEANS is not None:
    PAIRED_OUTFILE += '_kmeans'
    whitened = whiten(Xmotion.T)
    book, kmeans_labels = kmeans2(whitened, KMEANS, minit='points')
    # Compute coordinates in frame from centroids
    # seems to work better without sqrt
    dists = np.square(
        whitened[:, np.newaxis, :] - book[np.newaxis, :, :]).sum(-1)
    Xmotion = MIN_DIST / np.maximum(dists, MIN_DIST).T
    # Other solution: 1 for closest 1/2 for second, 0 for following
    clust_sort = dists.argsort(axis=1)
    Xmotion = np.zeros(Xmotion.shape)
    for i in range(Xmotion.shape[1]):
        Xmotion[clust_sort[i, -1], i] = 1.
        Xmotion[clust_sort[i, -2], i] = .5


print("Writting: Xsound %s, Xmotion %s, Y %s"
        % (paired_Xsound.shape, paired_Xmotion.shape, Y.shape))
# Write
OUT = OUT_DIR + PAIRED_OUTFILE + OUT_EXT
savemat(OUT,
        {
            'Xsound': paired_Xsound,
            'Xmotion': sp.csr_matrix(paired_Xmotion),
            'Y': Y,
            'labels': [l for l in labels],
            'sound_label_names': sound_label_names,
            }
        )
print "written: %s" % OUT
