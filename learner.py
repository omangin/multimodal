# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '01/2013'

"""Helper class to evaluate learners from multiple modalities.
"""


import numpy as np
from sklearn.decomposition.nmf import KLdivNMF, _normalize_sum

from lib.utils import safe_hstack


def learn_dictionary(data, k, iter_nmf=100, verbose=False):
    nmf = NMF(n_components=k, max_iter=iter_nmf, tol=0, init=None)
    nmf.fit(data, scale_W=True)
    return nmf


def fit_coefficients(data_obs, dictionary, iter_nmf=100, verbose=False):
    nmf_obs = NMF(n_components=dictionary.shape[0], max_iter=iter_nmf,
            tol=0, init=None)
    nmf_obs.components_ = dictionary
    coefficients = nmf_obs.transform(data_obs, scale_W=True)
    return coefficients


# NMF class with custom init
class NMF(KLdivNMF):

    def _init(self, X):
        n_samples, n_features = X.shape
        if self.init is None:
            H_init = _normalize_sum(np.abs(np.random.random(
                (self.n_components, n_features))) + .01, axis=1)
        else:
            assert(self.init.shape == (self.n_components, n_features))
            H_init = self.init
        W_init = X.dot(H_init.T)
        return W_init, H_init

    def transform(self, *args, **kwargs):
        self.init = self.components_
        return KLdivNMF.transform(self, *args, **kwargs)


class MultimodalLearner(object):

    def __init__(self, modalities, dimensions, coefficients, k,
                 sparseness=None, sp_coef=.1):
        self.mod = modalities  # Names of the modalities
        self.dim = dimensions  # Dimensions of modalities
        self.coef = coefficients  # Coefficients used to compensate
        self.k = k
        self.sparseness = sparseness  # data, components, None
        self.sp_coef = sp_coef
        # Eventually remove dimensions, as it is just used for debugging

    def train(self, data_matrices, iterations):
        n_samples = data_matrices[0].shape[0]
        for m, d in zip(data_matrices, self.dim):
            assert(m.shape == (n_samples, d))
        Vtrain = self.stack_data(self.mod, data_matrices)
        # Perform the experiment
        self.nmf_train = NMF(n_components=self.k, max_iter=iterations, tol=0,
                              init=None, sparseness=self.sparseness,
                              sp_coef=self.sp_coef)
        self.nmf_train.fit(Vtrain, scale_W=True)

    def get_dico(self, modality=None):
        dico = self.nmf_train.components_
        if modality is None:
            return dico
        else:
            start, stop = self.get_axis_range(modality)
            return dico[:, start:stop]

    def get_stacked_dicos(self, modalities):
        return safe_hstack([self.get_dico(modality=m) for m in modalities])

    def stack_data(self, modalities, data_matrices):
        coefs = [self.coef[self.get_index(mod)] for mod in modalities]
        return safe_hstack([c * m
                            for m, c in zip(data_matrices, coefs)])

    def get_axis_range(self, modality):
        idx = self.get_index(modality)
        start = sum(self.dim[:idx])
        stop = start + self.dim[idx]
        return (start, stop)

    def get_index(self, modality):
        return self.mod.index(modality)

    def reconstruct_internal(self, orig_mod, test_data, iterations):
        return self.reconstruct_internal_multi([orig_mod], [test_data],
                                                iterations)

    def reconstruct_internal_multi(self, orig_mods, test_data, iterations):
        for mod, data in zip(orig_mods, test_data):
            assert(data.shape[1] == self.dim[self.get_index(mod)])
        stacked_dico = self.get_stacked_dicos(orig_mods)
        stacked_data = self.stack_data(orig_mods, test_data)
        internal = fit_coefficients(stacked_data, stacked_dico,
                                    iter_nmf=iterations)
        return internal

    def reconstruct_modality(self, dest_mod, internal):
        return internal.dot(self.get_dico(dest_mod))

    def reconstruct_modalities(self, dest_mods, internal):
        return internal.dot(self.get_stacked_dicos(dest_mods))

    def modality_to_modality(self, orig_mod, dest_mod, test_data, iterations):
        return self.modalities_to_modalities([orig_mod], [dest_mod],
                                             [test_data], iterations)

    def modalities_to_modalities(self, orig_mods, dest_mods, test_data,
                                 iterations):
        internal = self.reconstruct_internal_multi(orig_mods, test_data,
                                                iterations)
        return self.reconstruct_modalities(dest_mods, internal)
