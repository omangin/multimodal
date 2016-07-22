# encoding: utf-8


"""Class to abstract learning from multiple modalities."""


from .lib.nmf import KLdivNMF as NMF
from .lib.array_utils import safe_hstack, GrowingLILMatrix


def fit_coefficients(data_obs, dictionary, iter_nmf=100, verbose=False):
    nmf_obs = NMF(n_components=dictionary.shape[0], max_iter=iter_nmf, tol=0)
    nmf_obs.components_ = dictionary
    coefficients = nmf_obs.transform(data_obs, scale_W=True)
    return coefficients


class MultimodalLearner(object):

    def __init__(self, modalities, dimensions, coefficients, k,
                 sparseness=None, sp_coef=.1):
        self.mod = modalities  # Names of the modalities
        self.dim = dimensions  # Dimensions of modalities
        self.coef = coefficients  # Coefficients used to compensate
                                  # between modalities
        self.k = k
        self.sparseness = sparseness  # data, components, None
        self.sp_coef = sp_coef
        self.dico = None  # None means not trained yet

    def train(self, data_matrices, iterations):
        n_samples = data_matrices[0].shape[0]
        for m, d in zip(data_matrices, self.dim):
            assert(m.shape == (n_samples, d))
        Vtrain = self.stack_data(self.mod, data_matrices)
        # Perform the experiment
        if self.sparseness is not None:
            raise NotImplemented
        self.nmf_train = NMF(n_components=self.k, max_iter=iterations, tol=0)
        self.nmf_train.fit(Vtrain, scale_W=True)
        self.dico = self.nmf_train.components_

    def get_dico(self, modality=None):
        if modality is None:
            return self.dico
        else:
            start, stop = self.get_axis_range(modality)
            return self.dico[:, start:stop]

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


class RollingLILMatrix(GrowingLILMatrix):
    """Grows until reaches number of rows, then replace rolling row.
    """

    def __init__(self, n_rows):
        GrowingLILMatrix.__init__(self)
        self._targer_n_rows = n_rows
        self._current_row = None

    def add_row(self, row):
        if self.shape[0] < self._targer_n_rows:
            GrowingLILMatrix.add_row(self, row)
        else:  # Replace current row
            if self._current_row is None:
                self._current_row = 0
            self[self._current_row, :] = row
            self._current_row = self._current_row % self._target_n_rows


# This needs some test and profiling....

class MiniBatchMultimodalLearner(object):

    def __init__(self, *args, **kwargs):
        n_batch = kwargs.pop('n_batch', 2 * kwargs['k'])
        batch_iter = kwargs.pop('mini_batch_iterations', 1)
        super(MiniBatchMultimodalLearner, self).__init__(*args, **kwargs)
        self.n_batch = n_batch
        self.batch_iter = batch_iter
        self.nmf_train = NMF(n_components=self.k, max_iter=self.batch_iter,
                             tol=0)
        self._initialized = False
        self._batch = [RollingLILMatrix(self.n_batch) for _ in self.modalities]

    @property
    def dico(self):
        return self.nmf_train.components_

    def train_on_row(self, rows_by_modality):
        init = not self._initialized
        self._initialized = True
        for i, r in enumerate(rows_by_modality):
            if not r.shape == (self.dim[i],):
                raise NotImplemented('Number of features has changed')
                # TODO: implement shape update on the current dictionary
            self._batch[i].add_row(r)
        Vtrain = self.stack_data(self.mod, rows_by_modality)
        # Perform the experiment
        if self.sparseness is not None:
            raise NotImplemented
        self.nmf_train.fit_update(Vtrain, scale_W=True, init=init,
                                  batch_iter=self.batch_iter)

    def train(self, data_matrices, iterations):
        raise NotImplemented(
            'Please use train_on_row for incremental learning.')
