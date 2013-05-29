# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '06/2012'


"""Module to convert frame representation as output by kinect recording
to angle and/or angle velocity representations.
"""


import numpy as np

import pyUtils.transformations as tf

from .utils import delayed_velocities, meta_map
from .vector_quantization import get_histos, kmeans, whiten
from .kde2d import gaussian_kde_2d

# Note: frame names from ros kinect seems to denote left / right from
#   the observer point of view.


ANGLES = [
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_hand'),
        ('torso', 'left_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_foot'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_hand'),
        ('torso', 'right_hip'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_foot'),
        ]


def angles_indices(marker_names):
    return [(marker_names.index(source), marker_names.index(dest))
            for source, dest in ANGLES]


def get_angles(sample, source_frame, dest_frame):
    """Compute rotation along three basis axis between two frames
    in the given sample.

    :param sample: array of translations and rotations (shape: (nb_frames, 7)
    :param source_frame, dest_frame: indices of source and dest frames
    """
    # All transformations are from the base frame, to get transformation from
    # one frame to the other, the first one needs to be inversed.
    # q = q1^{-1} * q2
    q = tf.quaternion_multiply(tf.quaternion_inverse(sample[source_frame, 3:]),
            sample[dest_frame, 3:])
    return tf.euler_from_quaternion(q)


def get_angle_array(sample, angles_idx):
    angles = [get_angles(sample, s, d) for s, d in angles_idx]
    return np.hstack(angles)


def record_to_angle_array(record, angles_idx):
    return np.vstack([get_angle_array(sample, angles_idx)
        for sample in record])


def db_to_list_of_angle_arrays(db):
    angle_idx = angles_indices(db.marker_names)
    return [record_to_angle_array(r[0], angle_idx) for r in db.records]


def db_to_angles_and_vels(db, vel_delay=1, vel_padding='zeros'):
    angles = db_to_list_of_angle_arrays(db)
    vels = [delayed_velocities(vel_delay, angle, padding=vel_padding)
            for angle in angles]
    return angles, vels


def get_bounds(vels):
    min_vel = np.min(np.vstack(vels))
    max_vel = np.max(np.vstack(vels))
    return min_vel, max_vel


def filter_values(data, bounds):
    """Filter big values in data, according to given bounds.
    :param data: numpy array
    :param bounds: (min, max)
    """
    cut = lambda x: np.maximum(np.minimum(x, bounds[1]), bounds[0])
    return map(cut, data)


def db_to_binned_hist_matrix(db, vel_delay=1, vel_padding='zeros',
        nb_bins=16, bounds=None, vel_bounds=None, rel_h=.3, fft=True):
    """Compute the histogram matrix from the database, using binned histograms
    smoothed by a Gaussian kernel.

    :param db:
        the Database

    :param vel_delay, vel_padding:
        delayed velocity parameters

    :param nb_bins: int,
        number of bins (output dimension of histograms for a joint)

    :param bounds, vel_bounds: (min, max), couples of floats
        bounds on angle and velocities, if given, data
        is cut to fit in bounds, else they are computed from data.

    :param rel_h: float,
        relative width of the Gaussian smoother

    :param fft: bool,
        whether to use fft convolution (default)

    :return: (nb ex, nb features) matrix
    """
    angles, vels = db_to_angles_and_vels(db, vel_delay=1, vel_padding='zeros')
    # Angle bounds
    if bounds is None:
        bounds = get_bounds(angles)
    else:
        angles = filter_values(angles, bounds)
    # Velocity bounds
    if vel_bounds is None:
        vel_bounds = get_bounds(vels)
    else:
        vels = filter_values(vels, vel_bounds)
    # Histogram are specific to each angle and corresponding velocity
    # Compute Gaussian width from relative width for angles
    h = rel_h * (bounds[1] - bounds[0])
    # Compute gaussian width for velocities
    h_vel = rel_h * (vel_bounds[1] - vel_bounds[0])
    # For fair comparison with 1D hist and VQ
    nb_bins_sqrt = int(np.sqrt(nb_bins))
    to_gaussKDEs2 = lambda x: [  # x = (angles, vels)
            gaussian_kde_2d(
                np.hstack([x[0][:, dim][:, np.newaxis],
                    x[1][:, dim][:, np.newaxis]]),
                h, h_vel, nb_bins=nb_bins_sqrt,
                bounds=(np.array([bounds[0], vel_bounds[0]]),
                    np.array([bounds[1], vel_bounds[1]])),
                fft=fft)
            for dim in range(x[0].shape[1])]
    kdes = map(to_gaussKDEs2, zip(angles, vels))
    # Each kde is a triplet (x_grid, y_grid, bins)
    # Get and flatten histograms (second element of the couple)
    hist = meta_map(2, lambda  x: x[2].flatten())(kdes)
    data_matrix = np.vstack(map(np.hstack, hist))
    return data_matrix


def compact_examples(x):
    """Vertically stack list of array and returns stacked
    array and indices to un_compact it.
    """
    idx = [y.shape[0] for y in x]
    return np.vstack(x), list(np.cumsum(idx))


def un_compact_examples(v, idx):
    return [v[i:j, :]
            for i, j in zip([0] + idx[:-1], idx)]


def db_to_VQ_hist_matrix(db, vel_delay=1, vel_padding='zeros',
        nb_bins=16, bounds=None, vel_bounds=None, soft_vq=None):
    """Compute the histogram matrix from the database, using binned histograms
    smoothed by a Gaussian kernel.

    :param db:
        the Database

    :param vel_delay, vel_padding:
        delayed velocity parameters

    :param nb_bins: int,
        number of bins (output dimension of histograms for a joint)

    :param bounds, vel_bounds: (min, max), couples of floats
        bounds on angle and velocities, if given, data
        is cut to fit in bounds

    :param soft_vq:
        if not None (default) soft vector quantization parameter.
    """
    angles, vels = db_to_angles_and_vels(db, vel_delay=1, vel_padding='zeros')
    # Angle bounds
    if bounds is not None:
        angles = filter_values(angles, bounds)
    # Velocity bounds
    if vel_bounds is not None:
        vels = filter_values(vels, vel_bounds)
    # For each DOF and each example compute 2D angle-vel vects
    # angles / vels => [(time, dof) for each example]
    nb_dofs = angles[0].shape[1]
    nb_ex = len(angles)
    data = [[np.hstack([a[:, dof][:, np.newaxis],
                        v[:, dof][:, np.newaxis]])
             for a, v in zip(angles, vels)]
            for dof in range(nb_dofs)]
    compacted = map(compact_examples, data)
    # Whiten data for each dof
    all_data = [whiten(d) for d, _ in compacted]

    # Compute centroids for each DOF
    centro = [kmeans(d, nb_bins, iter=20)[0] for d in all_data]
    # Compute hitograms for each sample
    histos = [get_histos(d, c, soft=soft_vq)
              for d, c in zip(all_data, centro)]
    # Group and sum by example
    histos_by_ex = [un_compact_examples(h, c[1])
                    for h, c in zip(histos, compacted)]
    ex_histos = np.array([[h.sum(axis=0) for h in hs] for hs in histos_by_ex])
    # ex_histo is now (nb_dofs, nb_ex, nb_bins)
    Xdata = np.swapaxes(ex_histos, 0, 1).reshape((nb_ex, nb_bins * nb_dofs))
    Xdata /= Xdata.sum(axis=1)[:, np.newaxis]
    return Xdata
