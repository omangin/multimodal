# -*- coding: utf-8 -*-


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '02/2011'

"""Two dimensional kernel density estimation.
"""


import numpy as np
from scipy.signal import convolve2d, fftconvolve
from scipy.integrate import trapz


def get_bins_2d(samples, nb_bins, bounds=None):
    """Create a grid of nb_bins^2 bins and spread weights into bins
    in the following manner:
    - for n samples,
    - bins are from -x_min to (x_max + delta_x) for each dimension,
    - weights are spread between four neighbours in the same way as
      for one dimensional samples.

    @param samples: (n, 2) shaped array of samples,
    @param nb_bins: number of bins, must be at least 2.
    @param bounds: if supplied (not None (default)), in the form,
        uses (array[x_min, y_min], array[x_max, y_max]) = bounds to generate
        grid and bins.

    @returns (x_grid, y_grid, bins):
        - x_(/_y)grid: position of the left (/bottom) borders of bins,
        - bins: bins array containing the sum of weights over samples.
    """
    if nb_bins < 2:
        raise ValueError('Number of bins must be at least two, ' + str(nb_bins))
    (nb_samples, d) = samples.shape
    if not d == 2:
        raise ValueError('Samples must be two dimensional.')
    # Binning samples
    if bounds is None:
        x_min = np.min(samples[:, 0])
        x_max = np.max(samples[:, 0])
        y_min = np.min(samples[:, 1])
        y_max = np.max(samples[:, 1])
    else:
        x_min = bounds[0][0]
        y_min = bounds[0][1]
        x_max = bounds[1][0]
        y_max = bounds[1][1]
        # ? check bounds ?
    x_range = x_max - x_min
    y_range = y_max - y_min
    # Generate grid
    delta_x = x_range / (nb_bins - 1)
    delta_y = y_range / (nb_bins - 1)
    # See below for explaination of the +1 in the cardinal of the grid
    x_grid = x_min + (np.arange(nb_bins + 1) * delta_x)
    y_grid = y_min + (np.arange(nb_bins + 1) * delta_y)
    # Compute bins for samples
    x_bins = np.floor((samples[:, 0] - x_min) / delta_x)
    y_bins = np.floor((samples[:, 1] - y_min) / delta_y)
    weight = 1. / (nb_samples * delta_x * delta_y) # Common weighting factor
    # Populate bins
    # Since x_max goes to last bin but coincidate with last bin right border
    # a 0 weight have to be added to the bin after the last one. To avoid
    # getting an out of bound index, this bin is added here and removed
    # before returning weights.
    bins = np.zeros((nb_bins + 1, nb_bins + 1))
    for (i, (xb, yb)) in enumerate(zip(x_bins, y_bins)): # Loop over samples
        weight_left   = (x_grid[xb + 1] - samples[i, 0]) / delta_x
        weight_bottom = (y_grid[yb + 1] - samples[i, 1]) / delta_y
        weight_right  = (samples[i, 0] - x_grid[xb]) / delta_x
        weight_up     = (samples[i, 1] - y_grid[yb]) / delta_y
        bins[    xb,     yb] += weight * weight_left * weight_bottom
        bins[    xb, yb + 1] += weight * weight_left * weight_up
        bins[xb + 1,     yb] += weight * weight_right * weight_bottom
        bins[xb + 1, yb + 1] += weight * weight_right * weight_up
    # See previous comment for explanation of the [:-1]
    return (x_grid[:-1], y_grid[:-1], bins[:-1, :][:, :-1])

def gaussian_kde_2d(samples, h_x, h_y, nb_bins, fft=True, bounds=None):
    """Estimate probability from a two dimensional samples, using a Gaussian
    n kernel density estimator.

    @param samples: samples, given as a (n, 2) shaped numpy array,
    @param h: width of the Gaussian kernel,
    @param nb_bins: number of grid points to use by dimension,
    @param fft: whether to use FFT to compute convolution.
    @param bounds: if supplied (not None (default)), in the form,
        uses (array[x_min, y_min], array[x_max, y_max]) = bounds to generate
        grid and bins.

    @returns: (grid, kde) where kde is the bin grid and kde the density
        estimator on a grid twice as large as grid.
    """
    # No need to check dimension, if wrong ValueError raised by get_bins2d...
    (x_grid, y_grid, bins) = get_bins_2d(samples, nb_bins, bounds=bounds)
    x_range = x_grid[-1] - x_grid[0] # x_max - x_min
    y_range = y_grid[-1] - y_grid[0] # y_may - y_min
    delta_x = x_grid[1] - x_grid[0]
    delta_y = y_grid[1] - y_grid[0]
    # Generate kernel values
    # Eventually only generate values between 0 and 3 or 4 times h
    # (after <1.e-5) and use symetry...
    # Generate grid for kernel values
    x_ker_grid = np.arange( -x_range / 2., x_range / 2., delta_x)
    y_ker_grid = np.arange( -y_range / 2., y_range / 2., delta_y)
    (xkl,) = x_ker_grid.shape # length of the kernel grid on x dimension
    (ykl,) = y_ker_grid.shape # length of the kernel grid on y dimension
    # Compute x_i^2 + y_j^2 for all (i, j)
    x_square = np.dot(x_ker_grid.reshape((xkl, 1)) ** 2, np.ones((1, ykl)))
    y_square = np.dot(np.ones((xkl, 1)), y_ker_grid.reshape((1, ykl)) ** 2)
    exp_denom_x = - .5 / h_x ** 2
    exp_denom_y = - .5 / h_y ** 2
    ker = np.exp(exp_denom_x * x_square + exp_denom_y * y_square)
    # Normalize kernel
    # TODO : ??? normalize 'by axis'
    s = np.sum(ker)
    if s == 0:
        print('Warning: window is too short...')
    ker = ker / s
    # Do convolution
    if fft:
        kde = fftconvolve(bins, ker, mode='same')
    else:
        kde = convolve2d(bins, ker, mode='same')
    return (x_grid, y_grid, kde)

EPSILON = 1.e-14

def kde_entropy_gaussian_2d(samples, h, nb_bins=100, fft=True):
    """Uses Kernel Density Estimator with Gaussian kernel on two
    dimensional samples x and returns estimated entropy.

    @param x: samples, given as a (n, 2) shaped numpy array,
    @param h: width of the Gaussian kernel,
    @param nb_bins: number of grid points to use,
    @param fft: whether to use FFT to compute convolution.
    """
    (x_grid, y_grid, kde) = gaussian_kde_2d(samples, h, nb_bins=nb_bins, fft=fft)
    delta_x = x_grid[1] - x_grid[0]
    delta_y = y_grid[1] - y_grid[0]
    plogp = - kde * np.log(kde + EPSILON)
    # Integrate
    entropy = trapz(trapz(plogp, dx=delta_y, axis=1), dx=delta_x)
    return entropy

def kde_KL_divergence_2d(x, y, h_x, h_y, nb_bins=100, fft=True):
    """Uses Kernel Density Estimator with Gaussian kernel on two
    dimensional samples x and y and returns estimated Kullback-
    Leibler divergence.

    @param x, y: samples, given as a (n, 2) shaped numpy array,
    @param h: width of the Gaussian kernel,
    @param nb_bins: number of grid points to use,
    @param fft: whether to use FFT to compute convolution.
    """
    min_ = np.min(np.vstack([np.min(x, axis=0), np.min(y, axis=0)]), axis=0)
    max_ = np.max(np.vstack([np.max(x, axis=0), np.max(y, axis=0)]), axis=0)
    bounds_ = np.vstack((min_, max_))
    (x_grid, y_grid, kde_x) = gaussian_kde_2d(x, h_x, h_y,
            nb_bins=nb_bins,
            fft=fft,
            bounds=bounds_
            )
    (x_grid2, y_grid2, kde_y) = gaussian_kde_2d(y, h_x, h_y,
            nb_bins=nb_bins,
            fft=fft,
            bounds=bounds_
            )
    delta_x = x_grid[1] - x_grid[0]
    delta_y = y_grid[1] - y_grid[0]
    plogp = - kde_x * np.log((kde_x + EPSILON) / (kde_y + EPSILON))
    # Integrate
    div = trapz(trapz(plogp, dx=delta_x, axis=1), dx=delta_y, axis=0)
    return div
