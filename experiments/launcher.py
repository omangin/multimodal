#!/usr/bin/env python2


import os
import time
import argparse
from collections import OrderedDict

from joblib import status
from joblib.job import Job
from joblib.process import MultiprocessPool
from joblib.torque import TorquePool, has_qsub

from multimodal.experiment import (TwoModalitiesExperiment,
                                   ThreeModalitiesExperiment)
from multimodal.db.choreo2 import Choreo2Loader
from multimodal.db.acorns import Year1Loader as AcornsLoader
from multimodal.db.objects import ObjectsLoader
from multimodal.lib.logger import Logger


parser = argparse.ArgumentParser()
parser.add_argument('action',
                    choices=['prepare', 'run', 'resume', 'plot', 'status'])
parser.add_argument('-w', '--watch', action='store_true')
parser.add_argument('-l', '--launcher', default=None,
                    choices=['process', 'torque'])
parser.add_argument('-s', '--save', action='store_true',
                    help="Save figures (only for plot).")
parser.add_argument('--no-plot', action='store_true',
                    help="Do not plot figures (only for plot).")
parser.add_argument('--plot-config', default=None,
                    help="Plot configuration as matplotlibrc.")
parser.add_argument('--plot-dest', default=None,
                    help="Plot destination (only for saving plots).")
parser.add_argument('--plot-format', default='svg',
                    help="Plot format, should be accepted by matplotlib "
                         "(only for saving plots).")


WORKDIR = os.path.expanduser('~/work/data/results/multimodal/')
SCRIPT2 = os.path.join(os.path.dirname(__file__), 'two_modalities.py')
SCRIPT3 = os.path.join(os.path.dirname(__file__), 'three_modalities.py')
DEFAULT_FIG_DEST = os.path.expanduser(
    '~/work/doc/illus/results/multimodal/gen/')
DEFAULT_PLOT_PARAMS = {
    'font.family': 'serif',
    'font.size': 9.0,
    'font.serif': 'Computer Modern Roman',
    'text.usetex': 'True',
    'text.latex.unicode': 'True',
    'axes.titlesize': 'large',
    'axes.labelsize': 'large',
    'legend.fontsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'path.simplify': 'True',
    'savefig.bbox': 'tight',
    'figure.figsize': (7.5, 6),
}

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
           ThreeModalitiesExperiment(
               {'image': ObjectsLoader(['SURF', 'color']),
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
    pool = TorquePool(default_walltime=240)
pool.extend(jobs)


MOD_PAIRS = [('motion', 'sound'),
             ('image', 'motion'),
             ('image', 'sound')]
EXPS_BY_NAME = dict(exps)
JOBS_BY_NAME = dict({j.name: j for j in jobs})


def collect_results():
    loggers = {}
    for mod1, mod2 in MOD_PAIRS:
        loggers[(mod1, mod2)] = []
        for k in Ks:
            current_loggers = []
            for i in range(N_RUN):
                job = JOBS_BY_NAME["{}_{}_{}_{}".format(mod1, mod2, k, i)]
                log = Logger.load(os.path.join(job.path, job.name),
                                  load_np=False)
                current_loggers.append(log)
            loggers[(mod1, mod2)].append(
                Logger.merge_experiments(current_loggers))
    return loggers


def collect_results3():
    loggers = {}
    for mod1, mod2 in MOD_PAIRS:
        loggers[(mod1, mod2)] = []
    for k in Ks:
        current_loggers = []
        for i in range(N_RUN):
            job = JOBS_BY_NAME["image_motion_sound_{}_{}".format(k, i)]
            current_loggers.append(
                Logger.load(os.path.join(job.path, job.name), load_np=False))
        merged_logger = Logger.merge_experiments(current_loggers)
        for mod1, mod2 in MOD_PAIRS:
            loggers[(mod1, mod2)].append(merged_logger)
    return loggers


def collect_results_image():
    loggers = OrderedDict()
    for feats in descriptor_sets:
        current_loggers = []
        for i in range(N_RUN):
            name = "image_sound_feats_{}_{}".format('_'.join(feats), i)
            job = JOBS_BY_NAME[name]
            log = Logger.load(os.path.join(job.path, job.name), load_np=False)
            current_loggers.append(log)
        loggers[tuple(feats)] = Logger.merge_experiments(current_loggers)
    return loggers


def get_stats():
    return "{} ({})".format(pool.status,
                            ', '.join(["%s: %s" % x
                                       for x in pool.status_counts]))


def print_stats():
    print(get_stats())


def print_refreshed_stats():
    while pool.status < status.FAILED:
        print(get_stats())
        time.sleep(1)
    print(get_stats())


if ACTION == 'prepare':
    # Generate config files
    for (n, e), j in zip(exps, jobs):
        e.save_serialized_parameters(j.config)
elif ACTION == 'run':
    pool.run()
    if LAUNCHER == 'process' or args.watch:
        print_refreshed_stats()
elif ACTION == 'status':
    if args.watch:
        print_refreshed_stats()
    else:
        print_stats()
elif ACTION == 'plot':
    import matplotlib
    # Set matplotlib config
    if args.plot_config is None:
        plot_params = matplotlib.rc_params()
        plot_params.update(DEFAULT_PLOT_PARAMS)
    else:
        plot_params = matplotlib.rc_params_from_file(args.plot_config,
                                                     fail_on_error=True)
    plot_params['interactive'] = not args.no_plot
    matplotlib.rcParams = plot_params
    assert(plot_params == matplotlib.rcParams)
    assert(plot_params is matplotlib.rcParams)
    from multimodal.plots import (plot_2k_graphs, plot_boxes,
                                  plot_boxes_by_feats, plot_boxes_all_mods)
    LOGGERS_2 = collect_results()
    LOGGERS_3 = collect_results3()
    LOGGERS_IMAGE = collect_results_image()
    plot_params = DEFAULT_PLOT_PARAMS
    fig_2k = plot_2k_graphs([LOGGERS_2, LOGGERS_3], Ks,
                            title='Cosine', metric='_cosine')
    idx50 = Ks.index(50)
    fig_one2one = plot_boxes(
        {mods: LOGGERS_2[mods][idx50] for mods in LOGGERS_2},
        LOGGERS_3[('image', 'motion')][idx50])
    fig_feats = plot_boxes_by_feats(LOGGERS_IMAGE)
    fig_all_mod = plot_boxes_all_mods(LOGGERS_3[('image', 'motion')][idx50])
    if args.save:
        if args.plot_dest is None:
            figure_destination = DEFAULT_FIG_DEST
        else:
            figure_destination = os.path.expanduser(args.plot_dest)
        by_name = {'one_to_one_all_k': fig_2k,
                   'one_to_one_all': fig_one2one,
                   'image_feats': fig_feats,
                   'three_modalities': fig_all_mod,
                   }
        for name in by_name:
            path = os.path.join(figure_destination,
                                name + '.' + args.plot_format)
            fig = by_name[name]
            fig.savefig(path, transparent=True, bbox_inches='tight')
            print('Written: {}.'.format(path))
