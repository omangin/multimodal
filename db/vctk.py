# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '08/2012'


"""Corpus for the voice training toolkit from the Center for speech
technology research (University of Edinburgh).

Note:
-----
- speaker 280 is not referenced in speaker-info.txt
- speaker 315 does not have transcriptions
"""


import os

from ..local import CONFIG
from ..features.hac import build_codebooks_from_list_of_wav
from .models.vctk import VCTKDB


def default_vctk_dir():
    """May raise NoConfigValueError."""
    return os.path.join(CONFIG['db-dir'], 'VCTK-Corpus')


DEFAULT_VCTK_FILE = 'VCTK.json'


def load(db_file=None, blacklist=False):
    if db_file is None:
        db_file = os.path.join(default_vctk_dir(), DEFAULT_VCTK_FILE)
    db = VCTKDB()
    db.load_from(db_file)
    return db


def build_vctk_codebook(ks):
    db = load()
    return build_codebooks_from_list_of_wav(
        [r.get_audio_path() for r in db.all_records()], ks)
