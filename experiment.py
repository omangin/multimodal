from itertools import product

from .lib.logger import Logger
from .lib.metrics import kl_div, rev_kl_div, cosine_diff, frobenius
from .lib.utils import random_split
from .pairing import associate_labels
from .learner import MultimodalLearner
from .evaluation import classify_NN, found_labels_to_score, chose_examples


DEFAULT_PARAMS = {
    'acorns_speaker': 0,
    'iter_train': 50,
    'iter_test': 50,
    'k': 50,
    }


class Experiment(object):

    def __init__(self):
        self.logger = Logger()


TEST = 0
EX = 1

INTERNAL = -1


class TwoModalitiesExperiment(Experiment):

    n_modalities = 2

    def __init__(self, loaders, k, coefs, iter_train, iter_test, debug=False):
        super(TwoModalitiesExperiment, self).__init__()
        self.modalities = loaders.keys()
        self.loaders = [loaders[m] for m in self.modalities]
        self.k = k
        self.coefs = coefs
        self.iter_train = iter_train
        self.iter_test = iter_test
        self.debug = debug
        for attr in ['modalities', 'k', 'coefs', 'iter_train', 'iter_test',
                     'debug']:
            self.logger.store_global(attr, self.__getattribute__(attr))

    def load_data(self):
        raw_data = [loader.get_data() for loader in self.loaders]
        # Generate pairing
        raw_labels = [loader.get_labels() for loader in self.loaders]
        (label_assoc, labels, assoc_idx) = associate_labels(raw_labels)
        self.label_association = label_assoc
        self.labels_all = labels
        self.data = [x[[idx[i] for idx in assoc_idx], :]
                     for i, x in enumerate(raw_data)]
        self.logger.store_global('label-pairing', self.label_association)
        self.logger.store_global('sample-pairing', assoc_idx)
        if self.debug:  # Reduce size of data for quick execution
            self.logger.log(
                    'WARNING: Debug mode active, using subset of the database')
            self.data = [x[:200, :11] for x in self.data]
            self.labels_all = self.labels_all[:200]
        # Extract examples for evaluation
        self.examples = chose_examples([l for l in self.labels_all])
        self.logger.store_global('examples', self.examples)
        self.others = [i for i in range(len(self.labels_all))
                         if i not in self.examples]
        self.data_ex = [x[self.examples, :] for x in self.data]
        self.data = [x[self.others, :] for x in self.data]
        self.labels_ex = [self.labels_all[i] for i in self.examples]
        self.labels = [self.labels_all[i] for i in self.others]
        self.run_generator = random_split(self.n_samples, .1)
        # Safety...
        assert(set(self.labels_ex) == set(self.labels_all))
        assert(all([self.n_samples == x.shape[0] for x in self.data]))
        assert(all([l in range(10) for l in self.labels]))

    @property
    def n_features(self):
        return [x.shape[1] for x in self.data]

    @property
    def n_samples(self):
        return len(self.labels)

    def run(self):
        try:
            while True:
                self._perform_one_run()
        except StopIteration:
            pass

    def _perform_one_run(self):
        train, test = self.run_generator.next()
        self.logger.new_run()
        data_train = [x[train, :] for x in self.data]
        data_test = [x[test, :] for x in self.data]
        test_labels = [self.labels[t] for t in test]
        # Init Learner
        learner = MultimodalLearner(self.modalities, self.n_features,
                                    self.coefs, self.k)
        ## Train
        learner.train(data_train, self.iter_train)
        ## Test
        # Usage:
        # transformed_data_test/ex[original modality][destination modality]
        transformed_data_test = self._get_all_transformations(learner,
                                                              data_test)
        transformed_data_ex = self._get_all_transformations(learner,
                                                            self.data_ex)
        # For each combination of modalities:
        to_test = product(range(self.n_modalities),
                          range(self.n_modalities),
                          [-1] + range(self.n_modalities))
        for (mod1, mod2, mod_cmp) in to_test:
            for metric, suffix in zip(
                    [kl_div, rev_kl_div, frobenius, cosine_diff],
                    ['', '_bis', '_frob', '_cosine']):
                # Perform recognition
                found = classify_NN(transformed_data_test[mod1][mod_cmp],
                                    transformed_data_ex[mod2][mod_cmp],
                                    self.labels_ex, metric)
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

    def _get_all_internals(self, learner, data_set):
        return [learner.reconstruct_internal(self.modalities[mod],
                                             x, self.iter_test)
                for mod, x in enumerate(data_set)]

    def _get_score_key(self, mod1, mod2, mod_cmp, metric):
        return "score_{}2{}{}{}".format(
                self.modalities[mod1], self.modalities[mod2],
                '' if mod_cmp == INTERNAL else '_' + self.modalities[mod_cmp],
                metric)

    def _get_result(self, mod1, mod2, mod_cmp, metric_key):
        key = self._get_score_key(mod1, mod2, mod_cmp, metric_key)
        return self.logger.get_stats(key)

    def print_result_table(self):
        # Print result table
        width = 13
        table = " ".join(["{}" for _dummy in range(6)])
        print('-' * (width * 6 + 5))
        print('Modalities'.center(width * 3 + 2)
            + 'Score: avg (std)'.center(width * 3 + 2))
        print('-' * (width * 3 + 2) + ' ' + '-' * (width * 3 + 2))
        print table.format(*[s.center(width)
                            for s in ['Test', 'Reference', 'Comparison',
                                      'KL', 'Euclidean', 'Cosine']])
        print(' '.join(['-' * width] * 6))
        for mod1, mod2 in [(0, 1), (1, 0)]:
            for mod_comp, mod_comp_str in ([(-1, 'internal')]
                                           + list(enumerate(self.modalities))):
                mod_str = [self.modalities[mod1].center(width),
                           self.modalities[mod2].center(width),
                           mod_comp_str.center(width)]
                res_str = [("%.3f (%.3f)"
                            % self._get_result(mod1, mod2,
                                               mod_comp, metr))
                           for metr in ['', '_frob', '_cosine']]
                print table.format(*[s.center(width)
                                     for s in (mod_str + res_str)])
        print('-' * (width * 6 + 5))
