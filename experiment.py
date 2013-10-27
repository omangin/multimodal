from .db.acorns import load as load_acorns
from .db.acorns import load_features as load_acorns_features
from .db.choreo2 import load as load_choreo2
from .db.choreo2 import load_features as load_choreo2_features
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


class TwoModalitiesExperiment(Experiment):

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

        if self.debug:
            self.logger.log(
                    'WARNING: Debug mode active, using subset of the database')
            self.data = [x[:200, :11] for x in self.data]
            self.labels_all = self.labels_all[:200]

        # Extract examples for evaluation
        self.examples = chose_examples([l for l in self.labels_all])
        self.others = [i for i in range(len(self.labels_all))
                         if i not in self.examples]
        self.data_ex = [x[self.examples, :] for x in self.data]
        self.data = [x[self.others, :] for x in self.data]
        self.labels_ex = [self.labels_all[i] for i in self.examples]
        self.labels = [self.labels_all[i] for i in self.others]

        # Safety...
        assert(set(self.labels_ex) == set(self.labels_all))
        assert(all([self.n_samples == x.shape[0] for x in self.data]))
        assert(all([l in range(10) for l in self.labels]))

        self.run_generator = random_split(self.n_samples, .1)

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
        # Get internal coefs
        internal_test = [learner.reconstruct_internal(mod, x, self.iter_test)
                         for mod, x in zip(self.modalities, data_test)]
        internal_ex = [learner.reconstruct_internal(mod, x, self.iter_test)
                       for mod, x in zip(self.modalities, self.data_ex)]
        # Reconstruct one modality from an other
        Xreco_test = {}
        Xreco_ex = {}
        for (m1, m2) in [(0, 1), (1, 0)]:
            key = "%s_as_%s" % tuple(self.modalities[m] for m in (m1, m2))
            Xreco_test[key] = learner.reconstruct_modality(self.modalities[m2],
                                                           internal_test[m1])
            Xreco_ex[key] = learner.reconstruct_modality(self.modalities[m2],
                                                         internal_ex[m1])
        # Evaluate coefs
        to_test = []
        for (mod1, mod2) in [(0, 1), (1, 0)]:
            mod_str1 = self.modalities[mod1]
            mod_str2 = self.modalities[mod2]
            # Evaluate recognition in single modality
            to_test.append((mod_str1, internal_test[mod1], internal_ex[mod1]))
            # Evaluate one modality against the other:
            # - on internal coefficients
            to_test.append((mod_str1 + '2' + mod_str2,
                            internal_test[mod1], internal_ex[mod2]))
            # - original data for test mod1
            #   compared to reconstructed mod1 from mod2 reference examples
            to_test.append(("{}2{}_{}".format(mod_str1, mod_str2, mod_str1),
                           data_test[mod1],
                           Xreco_ex["%s_as_%s" % (mod_str2, mod_str1)]))
            # - reconstructed mod2 from mod1 test examples
            #   compared to original for mod2 reference examples
            to_test.append(("{}2{}_{}".format(mod_str1, mod_str2, mod_str2),
                           Xreco_test["%s_as_%s" % (mod_str1, mod_str2)],
                            self.data_ex[mod2]))

        for mod, coefs, coefs_ex in to_test:
            for metric, suffix in zip(
                    [kl_div, rev_kl_div, frobenius, cosine_diff],
                    ['', '_bis', '_frob', '_cosine']):
                # Perform recognition
                found = classify_NN(coefs, coefs_ex, self.labels_ex, metric)
                # Conpute score
                self.logger.store_result("score_%s%s" % (mod, suffix),
                                         found_labels_to_score(test_labels,
                                                               found))

    def _get_result(self, mod1, mod2, mod_comp, metric_key):
        key = "score_{}2{}{}{}".format(
                self.modalities[mod1], self.modalities[mod2],
                '' if mod_comp == 'internal' else '_' + mod_comp,
                metric_key)
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
        for mods in [(0, 1), (1, 0)]:
            for mod_comp in (['internal'] + self.modalities):
                mod_str = [self.modalities[mods[0]].center(width),
                           self.modalities[mods[1]].center(width),
                           mod_comp.center(width)]
                res_str = [("%.3f (%.3f)"
                            % self._get_result(mods[0], mods[1],
                                               mod_comp, metr))
                           for metr in ['', '_frob', '_cosine']]
                print table.format(*[s.center(width)
                                     for s in (mod_str + res_str)])
        print('-' * (width * 6 + 5))


# TODO: Maybe the loaders should move to datasets ?

class Choreo2Loader(object):

    def get_data(self):
        return load_choreo2_features()

    def get_labels(self):
        motion_db = load_choreo2(verbose=False)
        motion_names = motion_db.label_descriptions
        return [motion_names[r[1][0]] for r in motion_db.records]


class AcornsLoader(object):

    def __init__(self, speaker):
        self.speaker = speaker

    def get_data(self):
        return load_acorns_features(1, self.speaker)

    def get_labels(self):
        db = load_acorns(1, blacklist=True)
        return [db.tags[r.tags[0]] for r in db.records[self.speaker]]
