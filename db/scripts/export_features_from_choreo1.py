#!/usr/bin/python2
# -*- coding: utf-8 -*-


import os, sys
sys.path += [os.path.join(os.path.dirname(__file__), '../../')]

from scipy.io import savemat

from db.choreo1 import PRIMITIVES
from lib.skeleton import ANGLE_BOUNDS
from exp.exp.iros2012.helper_NMFexp import (
        Struct,
        load_data,
        transform_data,
        transform_data_velocities,
        transform_data_stft,
        )


#############################
#   Parameter definitions   #
#############################


PARAMS = {
        # Data files
        'file_pref' : '/home/omangin/work/data/capture/kinect/labeled/',
        'file_ext' : '.data',
        'file_names' : PRIMITIVES,
        # Subset of labels (if None, all labels are kept)
        'label_subset' : [2, 5, 11, 12, 13, 14, 15, 21, 22, 35],
        # Feature parameters
        'repr' : 'vel', # either 'normal', 'vel' 'stft'
        'vel_mode' : 'joint',
        'vel_delay' : 10,
        'vel_padding' : 'zeros',
        'histo_mode' : 'VQ',
        'nb_bins' : 15,
        # Relative width of Gaussian smoother (to be multiplied by bin width)
        # For KDE only
        'rel_h' : .3, # In proportion of the range
        # Soft or hard association (for VQ only)
        'VQ_alpha' : None, # Param for soft VQ, if None, hard VQ is used
        }

OUTFILE = '/home/omangin/work/data/capture/features_choreo1_primitives.mat'


######################
#   Run experiment   #
######################


if __name__ == '__main__':
    param_file = None

    # Easier access to parameters
    CONF = Struct(**PARAMS)

    # Load and group data
    print('Loading data...')
    (DATA, LABELS, NB_EXAMPLES, ALL_JOINTS) = load_data(CONF.file_pref,
            CONF.file_ext, CONF.file_names)
    print('Loaded: %d examples in %d files.' %(NB_EXAMPLES, len(CONF.file_names)))

    # Eventually filter labels
    if CONF.label_subset is not None:
        def contains(l, val):
            try:
                _ = l.index(val)
                return True
            except ValueError:
                return False
        filt_data = []
        filt_labels = []
        for (d, l) in zip(DATA, LABELS):
            if contains(CONF.label_subset, l):
                filt_data.append(d)
                filt_labels.append(l)
        DATA = filt_data
        LABELS = filt_labels
        NB_EXAMPLES = len(DATA)
        print('Filtered data: %d examples.' %(NB_EXAMPLES))

    # Extract features
    print('Transforming data:')
    if CONF.repr == 'vel':
        V_data = transform_data_velocities(
                    DATA, ANGLE_BOUNDS, CONF.nb_bins, CONF.rel_h,
                    VQ_alpha=CONF.VQ_alpha, mode=CONF.histo_mode,
                    vel_mode=CONF.vel_mode, vel_delay=CONF.vel_delay,
                    vel_padding=CONF.vel_padding
                    )
    elif CONF.repr == 'normal':
        V_data = transform_data(DATA, ANGLE_BOUNDS, CONF.nb_bins, CONF.rel_h,
                VQ_alpha=CONF.VQ_alpha, mode=CONF.histo_mode)
    elif CONF.repr == 'stft':
        V_data = transform_data_stft(
            DATA, CONF.stft_win, CONF.stft_hop, CONF.nb_bins)
    else:
        raise ValueError('Wrong representation mode.')
    data_dim = V_data.shape[1]

    # Get labels and count their occurences
    LABEL_SET = list(set(LABELS))
    NB_LABELS = len(LABEL_SET)
    LABEL_COUNTS = [LABELS.count(l) for l in LABEL_SET]
    print('  %d labels, average %d examples by label (min: %d, max: %d)' %(
        NB_LABELS,
        sum(LABEL_COUNTS) / len(LABEL_COUNTS),
        min(LABEL_COUNTS),
        max(LABEL_COUNTS))
        )

    # Data matrix is saved in column sample convention
    savemat(OUTFILE, {'Vdata': V_data.T, 'labels': LABELS, 'labels_subset': CONF.label_subset})
