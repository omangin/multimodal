# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '08/2012'


"""Choerography dataset v2.

    More information can be found at:
    http://flowers.inria.fr/choreo2/
"""


import os

from multimodal.local import CONFIG
from multimodal.db.models.kinect_motion import MotionDatabase


def default_file():
    return os.path.join(CONFIG['db-dir'], 'choreo2', 'choreo2.json')


## Load motion DB and features

def load(db_file=None, verbose=False):
    if db_file is None:
        db_file = default_file()
    return MotionDatabase.load_from_npz(db_file, verbose=verbose)

load.__doc__ = MotionDatabase.load_from_npz.__doc__
