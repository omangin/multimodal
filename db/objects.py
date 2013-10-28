# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '09/2013'


"""Objects dataset, provided by Natalia Lyubova.
   (www.ensta-paristech.fr/~lyubova).

   This dataset was recorded from demonstations of objects to an iCub
   robot with kinect vision. It is made of consecutive frame of
   the demonstrated object, in various positions.

   The helpers below only load examples with a single label.
"""


import os

from .models.objects import ObjectDB
from .models.loader import Loader
from ..local import CONFIG
from ..lib.array_utils import safe_hstack


DEFAULT_LABELS = [
    'blue octopus',
    'red pooh',
    'pink octopus',
    'yellow car',
    'blue and yellow whale',
    'blue-eyes-green-yellow',
    'orange fish',
    'brown and white squirrel',
    'mouse?',
    'dark green and white cube',
    ]


def get_default_db_file(extension=True):
    """May raise NoConfigValueError."""
    return os.path.join(CONFIG['db-dir'],
                        'objects',
                        'single_object_label' + ('.json' if extension else ''))


def load(db_file=None):
    if db_file is None:
        db_file = get_default_db_file()
    return ObjectDB.load(db_file)


class ObjectsLoader(Loader):

    def __init__(self, features, labels=DEFAULT_LABELS):
        super(ObjectsLoader, self).__init__()
        self.feature_list = features
        self._db = None
        self.keep_idx = None
        self.labels_to_keep = set(labels)

    @property
    def db(self):  # Lazzy loading of db
        if self._db is None:
            self._db = load()
            # Filter indices
            self.keep_idx = [i for i, f in enumerate(self._db.frames)
                             if self._db.object_names[f.label]
                                in self.labels_to_keep]
        return self._db

    def get_data(self):
        X = safe_hstack([self.db.get_histos_matrix_by_frame(f)
                         for f in self.feature_list])
        X = X.tocsr()[self.keep_idx, :]
        self.check_n_samples(X.shape[0])
        return X

    def get_labels(self):
        labels = [self.db.object_names[self.db.frames[i].label]
                  for i in self.keep_idx]
        self.check_n_samples(len(labels))
        return labels


def load_features_and_labels(features, db_file=None):
    db = load(db_file=db_file)
    labels = [f.label for f in db.frames]
    feats = load_features(features, db_file=None, db=db)
    return feats, labels, db.object_names


def load_features(features, db_file, db=None):
    if db is None:
        db = load(db_file=db_file)
    return safe_hstack([db.get_histos_matrix_by_frame(f) for f in features])
