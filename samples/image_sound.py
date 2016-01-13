#!/usr/bin/env python


"""Experiment for image and sound corresponding to icdl2013.py.
"""


import sys

from multimodal.experiment import TwoModalitiesExperiment
from multimodal.db.objects import ObjectsLoader
from multimodal.db.acorns import Year1Loader as AcornsLoader


DEBUG = False
if len(sys.argv) > 1 and sys.argv[1] == '--debug':
    DEBUG = True
    sys.argv.pop(1)

exp = TwoModalitiesExperiment({'objects': ObjectsLoader(['SURF', 'color']),
                               'sound': AcornsLoader(1)},
                              50, 50, 50, debug=DEBUG)
exp.run()
exp.print_result_table()
