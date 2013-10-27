#!/usr/bin/env python2


import sys

from multimodal.experiment import (TwoModalitiesExperiment, Choreo2Loader,
                                   AcornsLoader)


DEBUG = False
if len(sys.argv) > 1 and sys.argv[1] == '--debug':
    DEBUG = True
    sys.argv.pop(1)

exp = TwoModalitiesExperiment({'motion': Choreo2Loader(),
                                'sound': AcornsLoader(1)},
                                50, [1., 0.0008], 50, 50, debug=DEBUG)
exp.load_data()
exp.run()
exp.print_result_table()
