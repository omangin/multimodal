#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import sys
try:
    from termcolor import colored
except ImportError:
    def colored(s, *args, **kwargs):
        return s

from audio_tools.db.acornsDB import AcornsDB
from audio_tools.db.database import DONE


ACORNS_DIR = '/home/omangin/work/data/db/ACORNS/'
#ACORNS_DIR = '/Users/omangin/work/data/db/ACORNS/'
DB_ROOT = ACORNS_DIR + 'ACORNS-English-Corpora-ev01/ACORNS-Y2-ENG/'
# Default saving location
DB_DESCR = ACORNS_DIR + 'AcornsY2.xml'

# Name of the directories containing by-speaker db
SPEAKERS = ['Speaker-%.2d' %(i + 1)  for i in range(10)]


if len(sys.argv) > 1:
    out_file = sys.argv[1]
else:
    out_file = DB_DESCR
orig_db = AcornsDB()
orig_db.from_ACORNS_root(DB_ROOT, SPEAKERS)
orig_db.write_xml(out_file)
print(DONE + 'Saving to description file: %s.'
        %(colored(out_file, 'cyan')))
#db = AcornsDB()
#db.load_from(out_file)
