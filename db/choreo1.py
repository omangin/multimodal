# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '08/2012'


"""Choerography dataset. More information can be found at:
    http://flowers.inria.fr/choreography_database.html
"""


import os
import json

import numpy as np

from ..local import CONFIG


# Names of origin data files (for legacy)
# Primitives only db
PRIMITIVES = [
            'primitives_jal',
            'primitives_arm1',
            'primitives_all',
            'primitives_left_jal_06.15',
            'primitives_right_06.15',
            ]
# Mixed database
MIXED_BASE = [
            'mixed_jal_06.15',
            'mixed_arm_leg_07.07_1',
            'mixed_arm_leg_07.07_2',
            'mixed_arm_leg_07.07_3',
            'mixed_arm_leg_07.07_4',
            ]
# Primitives only db as a mixed db
PRIM_AS_MIXED = [
            'primitives_jal_as_mixed',
            'primitives_arm1_as_mixed',
            'primitives_all_as_mixed',
            'primitives_left_jal_06.15_as_mixed',
            'primitives_right_06.15_as_mixed',
            ]
# Extension with all labels
MIXED_FULL = MIXED_BASE + [
            'mixed_arm_leg_08.31_full_1',
            'mixed_arm_leg_08.31_full_2',
            ]
# Extension with only a subset of labels
MIXED_PARTIAL = MIXED_BASE + [
            'mixed_arm_leg_08.31_partial_1',
            'mixed_arm_leg_08.31_partial_2',
            'mixed_arm_leg_08.31_partial_3',
            ]
LABEL_SUBSET = [1, 5, 6, 10, 19, 20, 21, 22, 23,
                24, 25, 26, 28, 30, 38, 40, 43]


# List of available datasets
DATASETS = ['primitive', 'mixed_partial', 'mixed_full']


def load(dataset, data_path=None, verbose=False):
    """Load and return choreography database v1.

    Parameters
    ----------
    :param dataset: in ['primitive', 'mixed_partial', 'mixed_full']
        Which dataset to load.
    :param data_path: path
        Path to the data.
    :param verbose: boolean

    Returns
    -------
    (data, labels, markers) N: nb of samples
        - data: list of N numpy array of shape (T, nb markers, 3)
        - labels: list of N lists of labels (for each sample)
            (samples may have many labels)
        - markers: list of marker names
    """
    if data_path is None:
        data_path = os.path.join(CONFIG['db-dir'], 'choreo')
    if dataset not in DATASETS:
        raise ValueError("Dataset should be one of %s" % DATASETS)
    filename = os.path.join(data_path, dataset + '.json')
    with open(filename, 'r') as meta_file:
        meta = json.load(meta_file)
    loaded_data = np.load(os.path.join(os.path.dirname(filename),
                          meta['data-file']))
    data = []
    labels = []
    for r in meta['records']:
        data.append(loaded_data[str(r['data-id'])])
        labels.append(r['labels'])
    print("Loaded %d examples for ``%s`` database."
          % (len(data), meta['name']))
    print("Each data example is a (T, %d, 3) array."
          % len(meta['marker-names']))
    print("The second dimension corresponds to markers:")
    print("\t- %s") % '\n\t- '.join(meta['marker-names'])
    return (data, labels, meta['marker-names'])
