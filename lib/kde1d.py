# -*- coding: utf-8 -*-


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '02/2011'

"""One dimensional kernel density estimation.
"""


import numpy as np
from scipy import convolve as convolve_1d
from scipy.integrate import trapz
from scipy.signal import fftconvolve


def get_bins_1d(x, nb_bins, bounds=None):
    """Create a grid of nb_bins bins and spread weights into bins
    in the following manner:
    - for n samples,
    - bins are from -x_min to (x_max + delta_x),
    - if sample x is between left = (x_min + i * delta_x) and
      right = (x_min + (i + 1) * delta_x), weight
      (right - x) / (delta_x * n) (resp. (x - left) / (delta_x * n)) is added
      to bin i (resp. i + 1).

    @param x: (n,) shaped array of samples,
    @param nb_bins: number of bins, must be at least 2.
    @param bounds: if supplied (not None (default)), in the form,
        uses (x_min, x_max) = bounds to generate grid and bins.

    @returns (x_min, x_max, delta_x, grid, bins):
        - grid: position of the left border of bins,
        - bins: bins array containing the sum of weights over samples.
    """
    if nb_bins < 2:
        raise ValueError('Number of bins must be at least two, ' + str(nb_bins))
    (nb_samples,) = x.shape
    # Binning samples
    if bounds is None:
        x_min = np.min(x)
        x_max = np.max(x)
    else:
        (x_min, x_max) = bounds
        # Just in case...
        if (x_min > np.min(x)) or (x_max < np.max(x)):
            raise ValueError('Wrong bounds.')
    x_range = x_max - x_min
    # Generate grid
    delta_x = x_range / float(nb_bins - 1)
    # See below for explaination of the +1 in the cardinal of the grid
    x_grid = x_min + (np.arange(nb_bins + 1) * delta_x)
    # Compute bins for samples
    x_bins = np.floor((x - x_min) / delta_x)
    weight = 1. / (nb_samples * delta_x) # Common weighting factor
    # Populate bins
    # Since x_max goes to last bin but coincidate with last bin right border
    # a 0 weight have to be added to the bin after the last one. To avoid
    # getting an out of bound index, this bin is added here and removed
    # before returning weights.
    bins = np.zeros((nb_bins + 1,))
    for (i, b) in enumerate(x_bins): # Loop over samples
        weight_left = weight * (x_grid[b + 1] - x[i]) / delta_x
        weight_right = weight * (x[i] - x_grid[b]) / delta_x
        bins[b] += weight_left
        bins[b + 1] += weight_right
    # See previous comment for explanation of the [:-1]
    return (x_grid[:-1], bins[:-1])
    
# TODO decide something about returned grid size
def gaussian_kde_1d(x, h, nb_bins, fft=True, bounds=None):
    """Estimate probability from one dimensional samples, using a Gaussian
    kernel density estimator.

    @param x: samples, given as a (n,) shaped numpy array,
    @param h: width of the Gaussian kernel,
    @param nb_bins: number of grid points to use,
    @param fft: whether to use FFT to compute convolution.

    @returns: (grid, kde) where kde is the bin grid and kde the density
        estimator on a grid twice as large as grid.
    """
    (x_grid, bins) = get_bins_1d(x, nb_bins, bounds=bounds)
    x_range = x_grid[-1] - x_grid[0] # x_max - x_min
    delta_x = x_grid[1] - x_grid[0]
    # Generate kernel values
    # Eventually only generate values between 0 and 3 or 4 times h
    # (after <1.e-5) and use symetry...
    # Generate grid for kernel values
    ker_grid = np.arange( -x_range / 2., x_range / 2., delta_x)
    exp_fact = - .5 / h ** 2
    ker = np.exp(exp_fact * (ker_grid ** 2))
    # Normalize kernel
    s = np.sum(ker)
    if s == 0:
        print('Warning: window is too short...')
    ker = ker / s
    # Do convolution
    if fft:
        kde = fftconvolve(bins, ker)
    else:
        kde = convolve_1d(bins, ker)
    return (x_grid, kde)

EPSILON = 1.e-14

def kde_entropy_gaussian_1d(x, h, nb_bins=1000, fft=True):
    """Uses Kernel Density Estimator with Gaussian kernel on one
    dimensional samples x and returns estimated entropy.

    @param x: samples, given as a (n,) shaped numpy array,
    @param h: width of the Gaussian kernel,
    @param nb_bins: number of grid points to use,
    @param fft: whether to use FFT to compute convolution.
    """
    (grid, kde) = gaussian_kde_1d(x, h, nb_bins=nb_bins, fft=fft)
    delta = grid[1] - grid[0]
    plogp = - kde * np.log(kde + EPSILON)
    # Integrate
    entropy = trapz(plogp, dx=delta)
    return entropy

def kde_KL_divergence_1d(x, y, h, nb_bins=1000, fft=True):
    """Uses Kernel Density Estimator with Gaussian kernel on one
    dimensional samples x and y and returns estimated Kullback-
    Leibler divergence.

    @param x, y: samples, given as a (n,) shaped numpy array,
    @param h: width of the Gaussian kernel,
    @param nb_bins: number of grid points to use,
    @param fft: whether to use FFT to compute convolution.
    """
    min_ = np.min([np.min(x), np.min(y)])
    max_ = np.max([np.max(x), np.max(y)])
    (grid, kde_x) = gaussian_kde_1d(x, h,
            nb_bins=nb_bins,
            fft=fft,
            bounds=(min_, max_)
            )
    (grid2, kde_y) = gaussian_kde_1d(y, h,
            nb_bins=nb_bins,
            fft=fft,
            bounds=(min_, max_)
            )
    delta = grid[1] - grid[0]
    plogp = - kde_x * np.log((kde_x + EPSILON) / (kde_y + EPSILON))
    # Integrate
    div = trapz(plogp, dx=delta)
    return div
