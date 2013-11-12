#!/usr/bin/env python2


import os
import time
import argparse
import subprocess

from joblib import status
from joblib.job import Job
from joblib.process import MultiprocessPool
from joblib.torque import TorquePool

from multimodal.experiment import (TwoModalitiesExperiment,
                                   ThreeModalitiesExperiment)
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


WORKDIR = os.path.expanduser('~/work/data/results/multimodal/')
SCRIPT2 = os.path.join(os.path.dirname(__file__), 'two_modalities.py')
SCRIPT3 = os.path.join(os.path.dirname(__file__), 'three_modalities.py')

args = parser.parse_args()
LAUNCHER = args.launcher
if LAUNCHER is None:
    LAUNCHER = 'torque' if has_qsub() else 'process'
ACTION = args.action

Ks = [5, 10, 15, 20, 30, 40, 50, 75, 100, 200]
N_RUN = 20

DEFAULT_PARAMS = {
    'debug': False,
    'shuffle_labels': True,
    'run_mode': 'single',
    }

exps_2 = []
exps_2 += [("motion_sound_{}_{}".format(k, i),
          TwoModalitiesExperiment({'motion': Choreo2Loader(),
                                   'sound': AcornsLoader(1)},
                                  k, 50, 50, **DEFAULT_PARAMS)
          ) for k in Ks for i in range(N_RUN)]
exps_2 += [("image_sound_{}_{}".format(k, i),
          TwoModalitiesExperiment({'image': ObjectsLoader(['SURF', 'color']),
                                  'sound': AcornsLoader(1)},
                                 k, 50, 50, **DEFAULT_PARAMS)
         ) for k in Ks for i in range(N_RUN)]
exps_2 += [("image_motion_{}_{}".format(k, i),
          TwoModalitiesExperiment({'image': ObjectsLoader(['SURF', 'color']),
                                   'motion': Choreo2Loader()},
                                  k, 50, 50, **DEFAULT_PARAMS)
         ) for k in Ks for i in range(N_RUN)]
exps_3 = [("image_motion_sound_{}_{}".format(k, i),
         ThreeModalitiesExperiment({'image': ObjectsLoader(['SURF', 'color']),
                                    'motion': Choreo2Loader(),
                                    'sound': AcornsLoader(1)},
                                   k, 50, 50, **DEFAULT_PARAMS)
        ) for k in Ks for i in range(N_RUN)]

image_features = ['SURF', 'color', 'SURF_pairs', 'color_pairs',
                  'color_triplets']
descriptor_sets = ([[f] for f in image_features]
                   + [['SURF', 'color'],
                      ['SURF_pairs', 'color_pairs'],
                      ['SURF_pairs', 'color_triplets'],
                      image_features]
                   )
exp_images = [("image_sound_feats_{}_{}".format('_'.join(descriptors), i),
              TwoModalitiesExperiment(
                  {'image': ObjectsLoader(descriptors),
                   'sound': AcornsLoader(1)},
                  50, 50, 50, **DEFAULT_PARAMS)
         ) for descriptors in descriptor_sets for i in range(N_RUN)]

exps = exps_2 + exps_3 + exp_images

jobs = [Job(WORKDIR, n, SCRIPT2) for n, e in exps_2]
jobs += [Job(WORKDIR, n, SCRIPT3) for n, e in exps_3]
jobs += [Job(WORKDIR, n, SCRIPT2) for n, e in exp_images]

if LAUNCHER == 'process':
    pool = MultiprocessPool()
elif LAUNCHER == 'torque':
    pool = TorquePool(default_walltime=4.)
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
