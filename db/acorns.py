# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '08/2012'


"""ACORNS  CAREGIVER database.
"""


from scipy.io import loadmat
import scipy.sparse as sp

from audio_tools.db.acornsDB import AcornsDB


DATA_DIR = '/home/omangin/work/data/'
ACORNS_DIR = DATA_DIR + 'db/ACORNS/'
DB_DESCR_Y1 = ACORNS_DIR + 'Acorns_Y1.xml'
DB_DESCR_Y2 = ACORNS_DIR + 'Acorns_Y2.xml'

HAC_FEATURES_FILES_Y1 = [ACORNS_DIR + "acorns_HAC_Y1_S%0.2d" % speaker + '.mat'
                for speaker in range(1, 5)]
HAC_FEATURES_FILES_Y2 = [ACORNS_DIR + "acorns_HAC_Y2_S%0.2d" % speaker + '.mat'
                for speaker in range(1, 11)]
BLACKLIST_Y1 = [[277]]  # Bad records (empty, etc.)
# TODO auto blacklist


def check_year(year):
    if year not in [1, 2]:
        raise(ValueError, "Wrong year version: %d (should be 1 or 2)" % year)


def load(year, DB_FILE=None):
    """year: 1 or 2
    """
    check_year(year)
    if DB_FILE is None:
        DB_FILE = DB_DESCR_Y1 if year == 1 else DB_DESCR_Y2
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
    feat_files = HAC_FEATURES_FILES_Y1 if year == 1 else HAC_FEATURES_FILES_Y2
    hac_mat = loadmat(feat_files[speaker])
    hacs = hac_mat['FFFF'][0]  # HAC representation of records from speaker 1
    # Compute sound data matrix
    Xsound = sp.vstack([h.T for h in hacs]).tocsr()
    # CSR format, shape: (n, b)
    return Xsound
