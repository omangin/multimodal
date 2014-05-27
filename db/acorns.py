# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '08/2012'


"""ACORNS  CAREGIVER database.
"""


import os

from scipy.io import loadmat
import scipy.sparse as sp

from ..local import CONFIG
from .models.loader import Loader
from .models.acorns import AcornsDB, check_year


BLACKLIST_Y1 = [[], [176], [], []]  # Bad records (no label, empty, etc.)
BLACKLIST_Y2 = [[]] * 10


def default_acorns_dir():
    """May raise NoConfigValueError."""
    return os.path.join(CONFIG['db-dir'], 'ACORNS')


def default_acorns_file(year):
    return os.path.join("Acorns_Y%d.xml" % year)


def check_speaker(year, speaker):
    n_speaker = AcornsDB.n_speakers(year)
    if speaker < 0 or speaker > n_speaker:
        raise(ValueError, "Wrong speaker: %d (should between 1 and %d)"
              % (year, n_speaker))


def load(year, db_file=None, blacklist=False):
    """year: 1 or 2
    """
    check_year(year)
    if db_file is None:
        db_file = os.path.join(default_acorns_dir(), default_acorns_file(year))
    db = AcornsDB()
    db.load_from(db_file)
    if blacklist:
        for recs, bl in zip(db.records,
                            BLACKLIST_Y1 if year == 1 else BLACKLIST_Y2):
            for r in bl:
                recs.pop(r)
    return db


class Year1Loader(Loader):

    dataset_name = 'acorns'

    def __init__(self, speaker):
        super(Year1Loader, self).__init__()
        self.speaker = speaker

    def get_data(self):
        X = load_features(1, self.speaker, blacklist=True)
        self.check_n_samples(X.shape[0])
        return X

    def get_labels(self):
        db = load(1, blacklist=True)
        records = db.records[self.speaker]
        self.check_n_samples(len(records))
        return [db.tags[r.tags[0]] for r in records]

    def serialize(self):
        return self.speaker

    @classmethod
    def get_loader(cls, spkr):
        return Year1Loader(spkr)


# Deprecated for year 1
def load_features_and_labels(year, speaker, blacklist=False, db_file=None):
    db = load(year, db_file=db_file, blacklist=blacklist)
    labels = [r.tags for r in db.records[speaker]]
    Xsound = load_features(year, speaker, blacklist=blacklist)
    assert(Xsound.shape[0] == len(labels))
    return Xsound, labels, db.tags


def load_features(year, speaker, blacklist=False):
    check_year(year)
    check_speaker(year, speaker)
    feat_file = os.path.join(CONFIG['feat-dir'],
                             "acorns_HAC_Y%d_S%0.2d_python.mat" % (year, 1 + speaker))
    Xsound = loadmat(feat_file)['hac']
    # CSR format, shape: (n, b)
    if blacklist:
        all_records = range(Xsound.shape[0])
        for r in (BLACKLIST_Y1 if year == 1 else BLACKLIST_Y2)[speaker]:
            all_records.pop(r)
        Xsound = Xsound[all_records, :]

    return Xsound
