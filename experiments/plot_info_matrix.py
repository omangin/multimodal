import os

import numpy as np
import matplotlib.pyplot as plt

from expjobs.job import Job

from multimodal.pairing import organize_by_values
from multimodal.experiment import ThreeModalitiesExperiment
from multimodal.lib.logger import Logger
from multimodal.lib.plot import pcolormesh
from multimodal.lib.metrics import mutual_information
from multimodal.lib.munkres import min_weight_perm
from multimodal.db.choreo2 import Choreo2Loader
from multimodal.db.acorns import Year1Loader as AcornsLoader
from multimodal.db.objects import ObjectsLoader


WORKDIR = os.path.expanduser('~/work/data/results/multimodal')
SCRIPT3 = os.path.join(os.path.dirname(__file__), 'three_modalities.py')
DEFAULT_FIG_DEST = os.path.expanduser(
    '~/work/doc/illus/results/multimodal/gen/mutual_info')
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
CMAP = plt.cm.Blues

K = 15
N_RUN = 20

DEFAULT_PARAMS = {
    'debug': False,
    'shuffle_labels': True,
    'run_mode': 'single',
    }

exps = [("image_motion_sound_{}_{}".format(K, i),
         ThreeModalitiesExperiment(
            {'image': ObjectsLoader(['SURF', 'color']),
             'motion': Choreo2Loader(),
             'sound': AcornsLoader(1)},
            K, 50, 50, **DEFAULT_PARAMS)
         ) for i in range(N_RUN)]

jobs = [Job(WORKDIR, n, SCRIPT3) for n, e in exps]

MOD_PAIRS = [('motion', 'sound'),
             ('image', 'motion'),
             ('image', 'sound')]
EXPS_BY_NAME = dict(exps)
JOBS_BY_NAME = dict({j.name: j for j in jobs})


def internal_histograms_by_label(internal_values, by_labels_idx):
    """Compute a histogram of coefficient values for the group of samples
    corresponding to each label.
    """
    value_range = (internal_values.min(), internal_values.max())
    return [np.histogram(internal_values[idx], bins=10, range=value_range)[0]
            for idx in by_labels_idx]


def _sum_all_but_i(a, i):
    a = np.array(a)
    mask = np.ones(a.shape, dtype=np.bool)
    mask[i, :] = False
    return (a * mask).sum(axis=0)


def mutual_information_by_label(internal_values, by_labels_idx):
    """Compute information matrix for the joint distribution
    over the internal coefficient values and the presence of the label,
    for each label.
    """
    hists = internal_histograms_by_label(internal_values, by_label)
    joint_hists = [np.vstack([hists[i], _sum_all_but_i(hists, i)])
                   for i in range(len(hists))]
    return [mutual_information(1. * h / h.sum()) for h in joint_hists]


def plot_mutual_info(m, labels):
    perm = min_weight_perm(-m)
    not_in_perm = [i for i in range(K) if i not in perm]
    permuted_m = np.hstack([m[:, perm], m[:, not_in_perm]])
    p = pcolormesh(permuted_m, xticklabels=perm + not_in_perm,
                   yticklabels=labels, cmap=CMAP)
    cb = plt.colorbar(p, drawedges=False)
    cb.outline.set_visible(False)


# Re-compute training coefficients for experiments
mutual_information_matrices = []
plots = []
for exp_idx, (name, exp) in enumerate(exps):
    # Do not write log file
    exp.logger.filename = None
    # Load logger
    job = JOBS_BY_NAME[name]
    logger = Logger.load(os.path.join(job.path, job.name))
    # Set up label and sample associations
    raw_labels = exp._get_raw_labels()
    label_assoc = logger.get_value('label-pairing')
    sample_assoc = logger.get_value('sample-pairing')
    # Get label index in "new" label list (get sample id in modality 0,
    # get label for this sample and then get id in label_assoc table).
    labels = [label_assoc[0].index(raw_labels[0][a[0]]) for a in sample_assoc]
    exp._set_labels_and_prepare_data(label_assoc, labels, sample_assoc)
    exp._set_data_and_labels_for_examples(logger.get_value('examples'))
    labels = exp.labels  # Get subset of labels without examples
    learner = exp._get_new_learner()
    learner.dico = logger.get_last_value('dictionary')
    data = exp.data
    # Compute coefficients from data
    print("Re-computing internal coefficients...")
    internal = learner.reconstruct_internal_multi(exp.modalities,
                                                  data,
                                                  exp.iter_test)
    # Compute mutual information for each pair of coef and label
    n_labels = len(label_assoc[0])
    by_label = organize_by_values(labels, nb_values=n_labels)
    mutual_information_matrices.append(np.array([
        mutual_information_by_label(internal[:, i], by_label)
        for i in range(K)
        ]).T)
    # Each matrix is of dim (K x n_labels)
    # Plot result to disk
    with plt.rc_context(rc=DEFAULT_PLOT_PARAMS):
        plt.interactive(False)
        fig = plt.figure()
        sound_labels = label_assoc[exp.modalities.index("sound")]
        plot_mutual_info(mutual_information_matrices[-1], sound_labels)
        for ext in ['svg', 'pdf', 'eps']:
            path = os.path.join(DEFAULT_FIG_DEST,
                                'mutual_info_{}.{}'.format(exp_idx, ext))
            fig.savefig(path, transparent=True)
            print('Written: {}.'.format(path))
