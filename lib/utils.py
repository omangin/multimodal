# -*- coding: utf-8 -*-

__author__ = 'Olivier Mangin'
__date__ = '03/2011'

"""Various (rather quick and dirty) filtering utility.
"""


import numpy as np


def manualy_normalize(bounds, x):
    """Normalize given array according to given bounds.

    @param bounds: (low, up)
    @returns: normalized array
    """
    (low, up) = bounds
    widths = up - low
    widths = widths + 1. * (widths == 0)
    y = (x - low) / widths
    return y


# Pre-treat data
# Bound values
def filter_bound(x, inf=0, sup=1023):
    """Filter values according to the given sup and inf bounds.
    """
    return np.maximum(np.minimum(x, sup), inf)


# Some iteration tools

def meta_map(n, f):
    """Returns a map function which acts at a deeper list level.
    For example meta_map(2, f)([[a, b], [c]]) = [[f(a), f(b)], [f(c)]],
    meta_map(1, f)(l) = map(f, l).
    """
    if n == 0:
        return f
    else:
        return (lambda l: map(meta_map(n - 1, f), l))


def distribute(x, values, level):
    """Distribute elements in values on nth level of nested lists.
    This creates an additional nested level of lists.

    Exemple:
        distribute([[[1,2], [1,3]], [[2,2]]], ['a', 'b'], 2)
        --> [[
              [[1,2,'a'], [1,2,'b']],
              [[1,3,'a'], [1,3,'b']]
             ],
            [
            [[2,2,'a'], [2,2,'b']]
            ]
            ]
    """
    if level == 0:
        return [x + [v] for v in values]
    else:
        return [distribute(y, values, level - 1) for y in x]


def flatten(nested, level):
    """nested contains several levels of nested lists.
    Returns flatten iterator lists at given level.
    """
    if level == 0:
        yield nested
    else:
        for x in nested:
            for y in flatten(x, level - 1):
                yield y


# Apply continuity filter
# Here we simply use a average over 3 values
def filter_cont(data):
    """Apply simple continuity filter to array by averaging over three
    consecutive values. Averaging is made along the first dimension.

    @param data: two dimensional array of values.
    """
    l1 = data[:, :]
    l1 = np.vstack((l1[0, :], l1, l1[-1, :]))
    l2 = l1[ :-2, :]
    l3 = l1[2:  , :]
    l1 = l1[1:-1, :]
    ll = (l1 + l2 + l3) / 3
    return ll


# Velocities extraction

def velocities(positions):
    """Computes differences between two time step and returns them.

    @param positions: two dimensional array.
    """
    slided = np.vstack((positions[1:, :], positions[0, :]))
    return slided - positions


def delayed_velocities(delay, positions, padding='circular'):
    """Computes differences between each position and the one at some delay
    in the past.
    For missing values, array is considered circular.

    @param delay: int.
    @param positions: two dimensional array of positions
    @param padding: either 'circular' (default) or 'zeros', indicates
        how to fill missing values.
    @returns: array of same dimension.
    """
    (n, d) = positions.shape
    if delay > n:
        delay = n
    if padding == 'circular':
        slided = np.vstack([positions[-delay:, :], positions[:-delay, :]])
    elif padding == 'zeros':
        slided = np.vstack([np.zeros((delay, d)), positions[:-delay, :]])
    else:
        raise ValueError("Wrong padding (either 'circular' or 'zeros')")
    return positions - slided


# Tools to split between train and test sets (for cross-validation, etc.)
def random_split(n_samples, ratio):
    nb_test = int(ratio * n_samples)
    # Generate train and test set
    indices = range(n_samples)
    np.random.shuffle(indices)
    for i in range(n_samples)[::nb_test]:
        test = indices[i:(i + nb_test)]
        train = indices[:i] + indices[(i + nb_test + 1):]
        yield train, test


def leave_one_out(n_samples):
    for i in range(n_samples):
        # Generate train and test set
        train = range(n_samples)
        train.remove(i)
        yield train, [i]
