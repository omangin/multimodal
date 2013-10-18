# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '08/2012'


"""ACORNS  CAREGIVER database.
"""


import os

from scipy.io import loadmat
import scipy.sparse as sp

from multimodal.local import CONFIG
from multimodal.db.models.acorns import AcornsDB, check_year


# TODO auto blacklist
BLACKLIST_Y1 = [[277]]  # Bad records (empty, etc.)


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


def load(year, DB_FILE=None):
    """year: 1 or 2
    """
    check_year(year)
    if DB_FILE is None:
        DB_FILE = os.path.join(default_acorns_dir(), default_acorns_file(year))
    db = AcornsDB()
    db.load_from(DB_FILE)
    return db


def load_features_and_labels(year, speaker):
    db = load(year)
    labels = [r.tags for r in db.records[speaker]]
    Xsound = load_features(year, speaker)
    assert(Xsound.shape[0] == len(labels))
    return Xsound, labels, db.tags


def load_features(year, speaker):
    check_year(year)
    check_speaker(year, speaker)
    feat_file = os.path.join(CONFIG['feat-dir'],
                             "acorns_HAC_Y%d_S%0.2d.mat" % (year, speaker))
    hac_mat = loadmat(feat_file)
    hacs = hac_mat['FFFF'][0]  # HAC representation of records from speaker 1
    # Compute sound data matrix
    Xsound = sp.vstack([h.T for h in hacs]).tocsr()
    # CSR format, shape: (n, b)
    return Xsound
