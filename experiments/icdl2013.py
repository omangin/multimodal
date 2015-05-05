#!/usr/bin/env python


"""Main experiment presented in:
    **Learning semantic components from sub symbolic multi modal perception**
    O. Mangin and P.Y. Oudeyer
    *Joint IEEE International Conference on Development and Learning
     and on Epigenetic Robotics (ICDL EpiRob)*, Osaka (Japan) (2013)
"""


import sys

from multimodal.experiment import TwoModalitiesExperiment
from multimodal.db.choreo2 import Choreo2Loader
from multimodal.db.acorns import Year1Loader as AcornsLoader


DEBUG = False
if len(sys.argv) > 1 and sys.argv[1] == '--debug':
    DEBUG = True
    sys.argv.pop(1)

exp = TwoModalitiesExperiment({'motion': Choreo2Loader(),
                               'sound': AcornsLoader(1)},
                              50, 50, 50, debug=DEBUG)
exp.run()
exp.print_result_table()
