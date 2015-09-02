"""Classes to run experiments in a consistent manner."""


import os
import json
from datetime import datetime
from itertools import product
from collections import OrderedDict

import numpy as np

from .lib.logger import Logger
from .lib.metrics import kl_div, rev_kl_div, cosine_diff, frobenius
from .lib.utils import random_split, leave_one_out
from .pairing import associate_samples
from .learner import MultimodalLearner
from .evaluation import classify_NN, found_labels_to_score, chose_examples


DEFAULT_PARAMS = {
    'acorns_speaker': 0,
    'iter_train': 50,
    'iter_test': 50,
    'k': 50,
    }

TEST = 0
EX = 1

INTERNAL = -1


class Experiment(object):

    def __init__(self):
        self.logger = Logger()

    def set_out_path_and_name(self, path, name):
        self.logger.filename = os.path.join(path, name)


class MultimodalExperiment(Experiment):

    def __init__(self, loaders, k, iter_train, iter_test, coefs=None,
                 shuffle_labels=False, run_mode=.1, debug=False):
        super(MultimodalExperiment, self).__init__()
        self.modalities = list(loaders.keys())
        self.loaders = [loaders[m] for m in self.modalities]
        self.k = k
        self.coefs = coefs
        self.iter_train = iter_train
        self.iter_test = iter_test
        self.shuffle_labels = shuffle_labels
        self.run_mode = run_mode  # (ratio (float)|leave-one-out|single-run)
        self.debug = debug

    @property
    def n_modalities(self):
        return len(self.modalities)

    def _parameters_to_dict(self):
        d = {}
        for attr in self.params_to_store():
            d[attr] = self.__getattribute__(attr)
        return d

    def _get_raw_labels(self):
        return [loader.get_labels() for loader in self.loaders]

    def _set_coefs(self, raw_data):
        if self.coefs is None:
            self.coefs = [1. / np.average(x.sum(axis=1)) for x in raw_data]

    def _set_labels_and_prepare_data(self, label_assoc, labels, assoc_idx):
        self.label_association = label_assoc
        self.labels_all = labels
        self.logger.store_global('label_pairing', self.label_association)
        self.logger.store_global('sample_pairing', assoc_idx)
        # Load data
        raw_data = [loader.get_data() for loader in self.loaders]
        # Eventually compute weighting coefficients for modalities
        self._set_coefs(raw_data)
        # Order data
        self.data = [x[[idx[i] for idx in assoc_idx], :]
                     for i, x in enumerate(raw_data)]
        if self.debug:  # Reduce size of data for quick execution
            self.logger.log(
                'WARNING: Debug mode active, using subset of the database')
            self.data = [x[:200, :11] for x in self.data]
            self.labels_all = self.labels_all[:200]

    def _set_data_and_labels_for_examples(self, examples):
        """Split data and labels between examples and other (for train & test).
        """
        self.examples = examples
        self.logger.store_global('examples', self.examples)
        self.others = [i for i in range(len(self.labels_all))
                       if i not in self.examples]
        self.data_ex = [x[self.examples, :] for x in self.data]
        self.data = [x[self.others, :] for x in self.data]
        self.labels_ex = [self.labels_all[i] for i in self.examples]
        self.labels = [self.labels_all[i] for i in self.others]

    def prepare(self):
        # Log parameters
        params = self._parameters_to_dict()
        for key in params:
            self.logger.store_global(key, params[key])
        # Generate pairing and order data
        raw_labels = self._get_raw_labels()
        (label_assoc, labels, assoc_idx) = associate_samples(
            raw_labels, shuffle=self.shuffle_labels)
        self._set_labels_and_prepare_data(label_assoc, labels, assoc_idx)
        # Chose examples for evaluation and split labels and data
        self._set_data_and_labels_for_examples(
                chose_examples([l for l in self.labels_all]))
        self.run_generator = self.get_generator()
        # Safety...
        assert(set(self.labels_ex) == set(self.labels_all))
        assert(all([self.n_samples == x.shape[0] for x in self.data]))
        assert(all([l in range(10) for l in self.labels]))

    def get_generator(self):
        if self.run_mode == 'leave-one-out':
            return leave_one_out(self.n_samples)
        elif self.run_mode == 'single':
            # Just take one split:
            return iter([next(random_split(self.n_samples, .1))])
        else:
            return random_split(self.n_samples, self.run_mode)

    @property
    def n_features(self):
        return [x.shape[1] for x in self.data]

    @property
    def n_samples(self):
        return len(self.labels)

    def run(self):
        self.prepare()
        self.logger.store_global('start_time', str(datetime.now()))
        try:
            while True:
                self._perform_one_run()
        except StopIteration:
            pass
        self.logger.store_global('end_time', str(datetime.now()))
        try:
            self.logger.save()
        except self.logger.NoFileError:
            print('Not saving logs: no destination was provided.')

    def _get_new_learner(self):
        return MultimodalLearner(self.modalities, self.n_features,
                                 self.coefs, self.k)

    def _perform_one_run(self):
        train, test = next(self.run_generator)
        self.logger.new_run()
        self.logger.store('train', train)
        self.logger.store('test', test)
        data_train = [x[train, :] for x in self.data]
        data_test = [x[test, :] for x in self.data]
        test_labels = [self.labels[t] for t in test]
        # Init Learner
        learner = self._get_new_learner()
        # Train
        learner.train(data_train, self.iter_train)
        self.logger.store('dictionary', learner.get_dico())
        # Test
        self._evaluate(learner, data_test, test_labels)

    def _evaluate(self, learner, test, labels):
        raise NotImplemented

    def _get_all_internals(self, learner, data_set):
        return [learner.reconstruct_internal(self.modalities[mod],
                                             x, self.iter_test)
                for mod, x in enumerate(data_set)]

    def serialize_parameters(self):
        params = self._parameters_to_dict()
        params['loaders'] = [(l.dataset_name, l.serialize())
                             for l in self.loaders]
        return params

    def save_serialized_parameters(self, destination):
        params = self.serialize_parameters()
        with open(destination, 'w+') as f:
            json.dump(params, f)

    @classmethod
    def params_to_store(cls):
        return ['modalities', 'k', 'coefs', 'iter_train', 'iter_test',
                'shuffle_labels', 'run_mode', 'debug']

    @classmethod
    def get_loader(cls, dataset, conf):
        if dataset == 'acorns':
            from .db.acorns import Year1Loader
            cls = Year1Loader
        elif dataset == 'choreo2':
            from .db.choreo2 import Choreo2Loader
            cls = Choreo2Loader
        elif dataset == 'objects':
            from .db.objects import ObjectsLoader
            cls = ObjectsLoader
        else:
            raise ValueError("Unknown dataset: %s!" % dataset)
        return cls.get_loader(conf)

    @classmethod
    def from_serialized(cls, serialized):
        loaders = OrderedDict()
        for m, (dataset, conf) in zip(serialized['modalities'],
                                      serialized['loaders']):
            loaders[m] = cls.get_loader(dataset, conf)
        return cls(loaders, serialized['k'], serialized['iter_train'],
                   serialized['iter_test'], coefs=serialized['coefs'],
                   shuffle_labels=serialized['shuffle_labels'],
                   run_mode=serialized['run_mode'], debug=serialized['debug'])

    @classmethod
    def load_from_serialized(cls, path_to_serialized):
        with open(path_to_serialized, 'r+') as f:
            d = json.load(f)
        return cls.from_serialized(d)


class TwoModalitiesExperiment(MultimodalExperiment):

    def _evaluate(self, learner, data_test, test_labels):
        # transformed_data_test/ex[original modality][destination modality]
        transformed_data_test = self._get_all_transformations(learner,
                                                              data_test)
        transformed_data_ex = self._get_all_transformations(learner,
                                                            self.data_ex)
        # For each combination of modalities:
        to_test = product(list(range(self.n_modalities)),
                          list(range(self.n_modalities)),
                          [-1] + list(range(self.n_modalities)))
        for (mod1, mod2, mod_cmp) in to_test:
            for metric, suffix in zip(
                    [kl_div, rev_kl_div, frobenius, cosine_diff],
                    ['', '_bis', '_frob', '_cosine']):
                # Perform recognition
                found = classify_NN(transformed_data_test[mod1][mod_cmp],
                                    transformed_data_ex[mod2][mod_cmp],
                                    self.labels_ex, metric)
                # Store found labels
                self.logger.store_result(
                    self._get_found_key(mod1, mod2, mod_cmp, suffix), found)
                # Conpute score
                self.logger.store_result(
                    self._get_score_key(mod1, mod2, mod_cmp, suffix),
                    found_labels_to_score(test_labels, found))

    def _get_all_transformations(self, learner, data_set):
        """Computes all transformations of data.
        Returns list of lists of lists such that:
            result[input modality][output modality]
        Does not transform data if not necessary.
        """
        # First compute internals
        internals = self._get_all_internals(learner, data_set)
        # Compute all transformations
        out = [[[] for _dum in self.modalities]
               for _my in self.modalities]
        for in_mod in range(self.n_modalities):
            out[in_mod][in_mod] = data_set[in_mod]  # Nothing to do.
            for out_mod in range(self.n_modalities):
                if out_mod != in_mod:
                    out[in_mod][out_mod] = learner.reconstruct_modality(
                        self.modalities[out_mod], internals[in_mod])
            out[in_mod].append(internals[in_mod])
        return out

    def _get_exp_key(self, mod1, mod2, mod_cmp, metric):
        return "{}2{}{}{}".format(
            self.modalities[mod1], self.modalities[mod2],
            '' if mod_cmp == INTERNAL else '_' + self.modalities[mod_cmp],
            metric)

    def _get_score_key(self, mod1, mod2, mod_cmp, metric):
        return "score_" + self._get_exp_key(mod1, mod2, mod_cmp, metric)

    def _get_found_key(self, mod1, mod2, mod_cmp, metric):
        return "found_" + self._get_exp_key(mod1, mod2, mod_cmp, metric)

    def _get_result(self, mod1, mod2, mod_cmp, metric_key):
        key = self._get_score_key(mod1, mod2, mod_cmp, metric_key)
        return self.logger.get_stats(key)

    def print_result_table(self):
        # Print result table
        width = 13
        table = " ".join(["{}" for _dummy in range(6)])
        print('-' * (width * 6 + 5))
        print('Modalities'.center(width * 3 + 2) +
              'Score: avg (std)'.center(width * 3 + 2))
        print('-' * (width * 3 + 2) + ' ' + '-' * (width * 3 + 2))
        print(table.format(*[s.center(width)
                             for s in ['Test', 'Reference', 'Comparison',
                                       'KL', 'Euclidean', 'Cosine']]))
        print(' '.join(['-' * width] * 6))
        for mod1, mod2 in [(0, 1), (1, 0)]:
            for mod_comp, mod_comp_str in ([(-1, 'internal')] +
                                           list(enumerate(self.modalities))):
                mod_str = [self.modalities[mod1].center(width),
                           self.modalities[mod2].center(width),
                           mod_comp_str.center(width)]
                res_str = [("%.3f (%.3f)"
                            % self._get_result(mod1, mod2,
                                               mod_comp, metr))
                           for metr in ['', '_frob', '_cosine']]
                print(table.format(*[s.center(width)
                                     for s in (mod_str + res_str)]))
        print('-' * (width * 6 + 5))


class ThreeModalitiesExperiment(MultimodalExperiment):
    """Targetted for three modalities.

    Same kind of experiment as TwoModalitiesExperiment but only compare
    on internal modality.
    Performs all comparison of the forms:
    - mod1 vs mod2
    - (mod1, mod2) vs mod3
    """

    def _evaluate(self, learner, data_test, test_labels):
        # transformed_data_test/ex[original modality]
        transformed_data_test = self._get_all_internals(learner, data_test)
        transformed_data_ex = self._get_all_internals(learner, self.data_ex)
        # For each combination of modalities:
        for (mods1, mods2) in self._tested_combinations():
            for metric, suffix in zip(
                    [kl_div, rev_kl_div, frobenius, cosine_diff],
                    ['', '_bis', '_frob', '_cosine']):
                # Perform recognition
                found = classify_NN(transformed_data_test[mods1],
                                    transformed_data_ex[mods2],
                                    self.labels_ex, metric)
                # Conpute score
                self.logger.store_result(
                    self._get_score_key(mods1, mods2, suffix),
                    found_labels_to_score(test_labels, found))

    def _tested_combinations(self):
        combinations = []
        combinations += [([mod1], [mod2]) for mod1 in range(self.n_modalities)
                         for mod2 in range(self.n_modalities) if mod1 != mod2]
        two_to_one = [([m for m in range(self.n_modalities) if m != mod],
                       [mod])
                      for mod in range(self.n_modalities)]
        combinations += two_to_one
        combinations += [(y, x) for (x, y) in two_to_one]
        return [(tuple(x), tuple(y)) for (x, y) in combinations]

    def _get_all_internals(self, learner, data_set):
        internals = {}
        for mods, rest in self._tested_combinations():
            if mods not in internals:
                internals[mods] = learner.reconstruct_internal_multi(
                    [self.modalities[m] for m in mods],
                    [data_set[m] for m in mods],
                    self.iter_test)
        return internals

    def _get_score_key(self, mods1, mods2, metric):
        return "score_{}2{}{}".format(
            '_'.join([self.modalities[m] for m in mods1]),
            '_'.join([self.modalities[m] for m in mods2]),
            metric)

    def _get_result(self, mods1, mods2, metric_key):
        key = self._get_score_key(mods1, mods2, metric_key)
        return self.logger.get_stats(key)

    def print_result_table(self):
        # Print result table
        width = 15
        table = " ".join(["{}" for _dummy in range(5)])
        print('-' * (width * 5 + 5))
        print('Modalities'.center(width * 2 + 2) +
              'Score: avg (std)'.center(width * 3 + 2))
        print('-' * (width * 2 + 1) + ' ' + '-' * (width * 3 + 2))
        print(table.format(*[s.center(width)
                             for s in ['Test', 'Reference',
                                       'KL', 'Euclidean', 'Cosine']]))
        print(' '.join(['-' * width] * 5))
        for mods1, mods2 in self._tested_combinations():
            mod_str = [' & '.join(self.modalities[m]
                                  for m in mods1).center(width),
                       ' & '.join(self.modalities[m]
                                  for m in mods2).center(width),
                       ]
            res_str = ["{:.3f} ({:.3f})".format(
                       self._get_result(mods1, mods2, metr))
                       for metr in ['', '_frob', '_cosine']]
            print(table.format(*[s.center(width)
                                 for s in (mod_str + res_str)]))
        print('-' * (width * 5 + 5))
