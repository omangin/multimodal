# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '09/2013'


"""Objects dataset, provided by Natalia Lyubova.
   (www.ensta-paristech.fr/~lyubova).

   This dataset was recorded from demonstations of objects to an iCub
   robot with kinect vision. It is made of consecutive frame of
   the demonstrated object, in various positions.
"""


import os

from .models.objects import ObjectDB, FEATURES_INDEX
from ..local import CONFIG
from ..lib.array_utils import safe_hstack


def get_default_db_file(extension=True):
    """May raise NoConfigValueError."""
    return os.path.join(CONFIG['db-dir'],
                        'objects',
                        'single_object_label' + ('.json' if extension else ''))


def load(db_file=None):
    if db_file is None:
        db_file = get_default_db_file()
    return ObjectDB.load(db_file)


def load_features_and_labels(features, db_file=None):
    db = load(db_file=db_file)
    labels = [f.label for f in db.frames]
    feats = load_features(features, db_file=None, db=db)
    return feats, labels, db.object_names


def load_features(features, db_file, db=None):
    if db is None:
        db = load(db_file=db_file)
    return safe_hstack([db.histos[FEATURES_INDEX[f]] for f in features])
