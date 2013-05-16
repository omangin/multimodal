#!/usr/bin/env python2
# encoding: utf-8


"""Export feature matrices for Choreo v2.

    Convention for data matrices: shape -> (nb_features, nb_examples)
"""


import os

import numpy as np
from scipy.io import savemat

#from ros_skeleton_recorder.src.database import Database as MotionDB
from skeleton_recorder.database import Database as MotionDB

from exp.lib.tf_to_angles import angles_indices, db_to_VQ_hist_matrix


DATA_DIR = '/home/omangin/work/data/'
# Motion DB
#MOTION_DB_FILE = DATA_DIR + 'capture/kinect2/primitive.json'
#MOTION_FEATS = DATA_DIR + 'capture/features_primitives.npz'
MOTION_DB_FILE = DATA_DIR + 'capture/kinect2/primitives_all.json'
MOTION_FEATS = DATA_DIR + 'capture/features_primitives2.npz'

#OUTFILE = DATA_DIR + '/capture/features_choreo2_primitives.mat'
OUTFILE = DATA_DIR + '/capture/features_choreo2_primitives2.mat'


#NB_BINS = 9
NB_BINS = 15

EPSILON = 1.e-8


## Load motion DB and features

motion_db = MotionDB.load_from_npz(MOTION_DB_FILE)
print "Loaded %d samples." % motion_db.size()
ANGLE_INDICES = angles_indices(motion_db.marker_names)
if os.path.exists(MOTION_FEATS):
    # Load existing file of pre-computed features
    Xmotion = np.load(MOTION_FEATS)['Xmotion']
    assert (Xmotion.shape ==
            ((3 * NB_BINS * len(ANGLE_INDICES)), len(motion_db.records)))
    print 'Motion features loaded.\n'
else:
    # Or generate them
    print 'Computing histograms...'
    Xmotion = db_to_VQ_hist_matrix(motion_db,
            vel_delay=10, nb_bins=NB_BINS)  # k=16 seems too big
    #Xmotion = db_to_binned_hist_matrix(motion_db,
    #        vel_delay=10, nb_bins=NB_BINS)
    Xmotion = Xmotion.T
    np.savez(MOTION_FEATS, Xmotion=Xmotion)
    print "Motion features generated and save to: %s.\n" % MOTION_FEATS

label_set = motion_db.get_occuring_labels()
labels = [r[1] for r in motion_db.records]


# Write
savemat(OUTFILE, {'Vdata': Xmotion, 'labels': labels})
print "written: %s" % OUTFILE
