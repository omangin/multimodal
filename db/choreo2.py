# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '08/2012'


"""Choerography dataset v2.

    Still under construction.
    Uses representation from ros_skeleton_recorder.
"""


from skeleton_recorder.database import Database


DATA_DIR = '/home/omangin/work/data/'
# Motion DB
MOTION_DB_FILE = DATA_DIR + 'capture/kinect2/primitives_all.json'

## Load motion DB and features


def load(db_file=MOTION_DB_FILE):
    return Database.load_from_npz(db_file)

load.__doc__ = Database.load_from_npz.__doc__
