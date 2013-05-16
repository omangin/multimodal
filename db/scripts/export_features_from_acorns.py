#!/usr/bin/env python2
# encoding: utf-8


import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat, savemat
from scipy.cluster.vq import kmeans2, whiten

from audio_tools.db.acornsDB import AcornsDB

MOTION_FEATURE_FILE = '/home/omangin/work/data/capture/features_primitives.mat'

OUT_DIR = '/home/omangin/work/data/db/'
OUT_EXT = '.mat'
SOUND_OUTFILE =  'labels_and_features_acorns_Y1'
PAIRED_OUTFILE = 'paired_features'

BLACKLIST = [[277]]  # Bad records (empty, etc.)
DATA_DIR = '/home/omangin/work/data/'
ACORNS_DIR = DATA_DIR + 'db/ACORNS/'
DB_DESCR = ACORNS_DIR + 'Acorns_Y1.xml'
HAC_FILE = ACORNS_DIR + 'HAC_acorns_Y1_S1' + '.mat'

L_COEF = 1.  # Language coefficient
BETA = 2

KMEANS = 15
#KMEANS = None

EPSILON = 1.e-8


def labels_to_matrix(labels, k, normalize=False):
    """Transform a list of list of labels (or cluster affectations) into an
    histogram.

    :normalize: if True also makes sums over lines sum to 1
    :returns: (n, k) matrix with n the number of examples and k the number
        of labels
    """
    mtx = np.zeros((len(labels), k))
    for i, l in enumerate(labels):
        for a in l:
            mtx[i, a] += 1
    if normalize:
        mtx /= np.sum(mtx, axis=1)[:, np.newaxis]
    return mtx


# Load DB and features
db = AcornsDB()
db.load_from(DB_DESCR)
hac_mat = loadmat(HAC_FILE)
hacs = hac_mat['FFFF'][0]  # HAC representation of records from speaker 1
# Compute sound data matrix
X = sp.vstack([h.T for h in hacs])
X = sp.csr_matrix(X.T)  # To CSR format, shape: (f, n)

# Select records from speaker 1
records = [r for (i, r) in enumerate(db.records[0])]

# Build labels matrix from database
labels = [r.tags for r in records]
Y = labels_to_matrix(labels, len(db.tags))  # shape: (nb_ex, nb_labels)
# Cut unused last label
Y = Y[:, :10]
Y = sp.csr_matrix(Y.T)  # shape: (nb_labels, nb_ex)


# Keep 20 examples of each label for training and 10 for testing
by_tags = [[] for _ in xrange(10)]
for (i, r) in enumerate(db.records[0]):
    by_tags[r.tags[0]].append(i)
for l in by_tags:
    np.random.shuffle(l)
train_set = sum([l[:20] for l in by_tags], [])
test_set = sum([l[20:30] for l in by_tags], [])

Y_train = Y[:, train_set]
X_train = X[:, train_set]

# Test on single label recovery from sound examples
X_test = X[:, test_set]
Y_test = Y[:, test_set]

savemat(OUT_DIR + SOUND_OUTFILE + OUT_EXT,
        {'Xtrain': X_train,
            'Xtest': X_test,
            'Ytrain': Y_train,
            'Ytest': Y_test})

# Load motion features
motion_data = loadmat(MOTION_FEATURE_FILE)
Xmotion = motion_data['Vdata'].T
MOTION_LABELS = motion_data['labels'][:, 0]
SOUND2MOTION_LABELS = motion_data['labels_subset'][:, 0]
MOTION2SOUND_LABELS = dict([(l, i) for i, l in enumerate(SOUND2MOTION_LABELS)])

motion_labels_unified = [MOTION2SOUND_LABELS[l] for l in MOTION_LABELS]


# Plot PCA of data
zou = Xmotion - np.mean(Xmotion, axis=0)[np.newaxis, :]
vals, vects = np.linalg.eig(np.cov(zou))
zde = np.real(np.dot(vects.T, zou))
import matplotlib.pyplot as plt
from pyUtils.plots import get_n_colors
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = get_n_colors(10)
ax.scatter(zde.T[:, 0], zde.T[:, 1], zde.T[:, 2], c=[colors[l] for l in motion_labels_unified])
plt.show()



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
                                  

# Copy set of sounds examples by tag
by_tags_copy = [l[:] for l in by_tags]
# Pair examples:
# For each motion chose a sound with same label and remove it
# from available examples.
paired_sound_indices = [by_tags_copy[l].pop() for l in motion_labels_unified]
# Build sound data matrix
Xsound = X[:, paired_sound_indices]

# Write
savemat(OUT_DIR + PAIRED_OUTFILE + OUT_EXT,
        {'Xsound': Xsound,
            'Xmotion': Xmotion,
            'acorns_idx': paired_sound_indices,
            'sound2motion_labels': SOUND2MOTION_LABELS,
            'sound_labels': motion_labels_unified,  # Name not really intuitive
            }
        )
