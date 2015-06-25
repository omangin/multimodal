# encoding: utf-8


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
from ..lib.net import check_destination_path, urlretrieve


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
SRC = 'http://olivier.mangin.com/data/objects/'


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

    dataset_name = 'objects'

    def __init__(self, features, labels=DEFAULT_LABELS):
        super(ObjectsLoader, self).__init__()
        self.feature_list = features
        self._db = None
        self._keep_idx = None
        self.labels_to_keep = set(labels)

    @property
    def db(self):  # Lazzy loading of db
        if self._db is None:
            self._db = load()
        return self._db

    @property
    def keep_idx(self):
        if self._keep_idx is None:
            # Filter indices
            self._keep_idx = [i for i, f in enumerate(self.db.frames)
                              if self.db.object_names[f.label]
                              in self.labels_to_keep]
        return self._keep_idx

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

    def serialize(self):
        return {'features': self.feature_list,
                'labels': list(self.labels_to_keep)}

    @classmethod
    def get_loader(cls, cfg):
        return ObjectsLoader(cfg['features'], labels=cfg['labels'])


def load_features_and_labels(features, db_file=None):
    db = load(db_file=db_file)
    labels = [f.label for f in db.frames]
    feats = load_features(features, db_file=None, db=db)
    return feats, labels, db.object_names


def load_features(features, db_file, db=None):
    if db is None:
        db = load(db_file=db_file)
    return safe_hstack([db.get_histos_matrix_by_frame(f) for f in features])


def download_meta_and_features(dest):
    file_ = get_default_db_file(False)
    default_dest = os.path.dirname(file_)
    name = os.path.basename(file_)
    if dest is None:
        dest = default_dest
    meta_file = check_destination_path(dest, name + '.json')
    urlretrieve(SRC + name + '.json', meta_file)
    features_file = check_destination_path(dest, name + '.mat')
    urlretrieve(SRC + name + '.mat', features_file)
