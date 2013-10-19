# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '08/2012'


"""Choerography dataset v2.

    More information can be found at:
    http://flowers.inria.fr/choreo2/
"""


import os

import numpy as np

from multimodal.local import CONFIG
from multimodal.db.models.kinect_motion import MotionDatabase
from multimodal.features.angle_histograms import db_to_VQ_hist_matrix


DEFAULT_NB_BINS = 15  # Used to build features


def default_db_file():
    return os.path.join(CONFIG['db-dir'], 'choreo2', 'choreo2.json')


def get_default_feature_file():
    return os.path.join(CONFIG['feat-dir'], 'choreo2_angle_hist.npz')


# Load motion DB and features

def load(db_file=None, verbose=False):
    if db_file is None:
        db_file = default_db_file()
    return MotionDatabase.load_from_npz(db_file, verbose=verbose)

load.__doc__ = MotionDatabase.load_from_npz.__doc__


def load_features_and_labels(db_file=None, feat_file=None):
    db = load(db_file=db_file)
    X = load_features(feat_file=feat_file)
    labels = [r[1] for r in db.records]
    assert(X.shape[0] == len(labels))
    return X, labels, db.label_descriptions


def load_features(feat_file=None):
    if feat_file is None:
        feat_file = get_default_feature_file()
    return np.load(feat_file)['Xmotion']


def build_features(db_file=None, feat_file=None, force=False,
                   nb_bins=DEFAULT_NB_BINS, verbose=False):
    """Build histograms of angle and velocities features.
       May take a few minutes.
    """
    if feat_file is None:
        feat_file = get_default_feature_file()
    if os.path.exists(feat_file) and not force:
        raise ValueError("Feature file exists at {}.".format(feat_file))
    db = load(db_file=db_file)
    # Generate features
    if verbose:
        print 'Computing histograms (may take a few minutes)...'
    X = db_to_VQ_hist_matrix(db, vel_delay=10, nb_bins=nb_bins)
    np.savez(feat_file, Xmotion=X)
    if verbose:
        print("Motion features generated and saved to: %s." % feat_file)
