#!/usr/bin/env python2


import os
import time
import argparse
import subprocess

from joblib import status
from joblib.job import Job
from joblib.process import MultiprocessPool

from multimodal.experiment import TwoModalitiesExperiment
from multimodal.db.choreo2 import Choreo2Loader
from multimodal.db.acorns import Year1Loader as AcornsLoader
from multimodal.db.objects import ObjectsLoader


parser = argparse.ArgumentParser()
parser.add_argument('action',
        choices=['prepare', 'run', 'resume', 'plot', 'status'])
parser.add_argument('-w', '--watch', action='store_true')
parser.add_argument('-l', '--launcher', default=None,
                    choices=['process', 'torque'])


def has_qsub():
    with open(os.devnull) as devnull:
        return subprocess.call(['which', 'qsub'], stdout=devnull,
                               stderr=devnull) == 0


DEBUG = True
WORKDIR = os.path.expanduser('~/work/data/results/multimodal/')
SCRIPT = os.path.join(os.path.dirname(__file__), 'two_modalities.py')

args = parser.parse_args()
LAUNCHER = args.launcher
if LAUNCHER is None:
    LAUNCHER = 'torque' if has_qsub() else 'process'
ACTION = args.action

Ks = [10, 20, 30, 40, 50, 75, 100, 200]


exps = []
exps += [("motion_sound_{}".format(k),
          TwoModalitiesExperiment({'motion': Choreo2Loader(),
                                   'sound': AcornsLoader(1)},
                                  k, 50, 50, debug=DEBUG)
          ) for k in Ks]
exps += [("image_sound_{}".format(k),
          TwoModalitiesExperiment({'image': ObjectsLoader(['SURF', 'color']),
                                  'sound': AcornsLoader(1)},
                                 k, 50, 50, debug=DEBUG)
         ) for k in Ks]
exps += [("image_motion_{}".format(k),
          TwoModalitiesExperiment({'image': ObjectsLoader(['SURF', 'color']),
                                   'motion': Choreo2Loader()},
                                  k, 50, 50, debug=DEBUG)
         ) for k in Ks]

jobs = [Job(WORKDIR, n, SCRIPT) for n, e in exps]

pool = MultiprocessPool()
pool.extend(jobs)


def get_stats():
    return "{} ({})".format(pool.status,
                            ', '.join(["%s: %s" % x
                                       for x in pool.status_counts]))


def print_stats():
    print get_stats()


def print_refreshed_stats():
    while pool.status < status.FAILED:
        print get_stats()
        time.sleep(1)
    print get_stats()


if ACTION == 'prepare':
    # Generate config files
    for (n, e), j in zip(exps, jobs):
        e.serialize_parameters(j.config)
elif ACTION == 'run':
    pool.run()
    if LAUNCHER == 'process' or args.watch:
        print_refreshed_stats()
elif ACTION == 'status':
    if args.watch:
        print_refreshed_stats()
    else:
        print_stats()
