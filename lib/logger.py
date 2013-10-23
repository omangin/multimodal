# -*- coding: utf-8 -*-


"""Tools for simplified and generic logging during experiments.
"""


import json
import numpy as np


class Logger(object):

    class NoFileError(Exception):
        pass

    def __init__(self, filename=None):
        self.glob = {}  # Global logs
        self.exps = []  # Logs of experiments
        self.exp_keys = set()
        self.result_keys = set()
        self.filename = filename

    # Does not handle correctly numpy arrays with several values
    def __eq__(self, other):
        return (isinstance(other, Logger)
                and self.filename == other.filename
                and self.glob == other.glob
                and self.exps == other.exps
                and self.exp_keys == other.exp_keys
                and self.result_keys == other.result_keys
                )

    def new_experiment(self):
        self.exps.append({})

    def store_global(self, key, value):
        self.glob[key] = value

    def store(self, key, value):
        if key in self.exps[-1]:
            raise(ValueError,
                  "There is already a value for this experiment for key: %s."
                  % key)
        else:
            self.exp_keys.add(key)
            self.exps[-1][key] = value

    def store_result(self, key, value, is_data=False):
        self.store(key, value)
        self.result_keys.add(key)

    def get_values(self, key):
        return [dic[key] if key in dic else None for dic in self.exps]

    def get_value(self, key):
        return self.glob[key]

    def get_last_value(self, key):
        return self.exps[-1][key]

    def get_stats(self, key):
        vals = self.get_values(key)
        return (np.average(vals), np.std(vals))

    def print_result(self, key, text=None):
        if text is None:
            text = key
        avg, std = self.get_stats(key)
        print("%s: %g (std dev.: %g)" % (text, avg, std))

    def print_all_results(self):
        for k in self.result_keys:
            self.print_result(k)

    def save(self, compress=True):
        if self.filename is None:
            raise(self.NoFileError, 'No file set for this logger.')
        else:
            to_save = {'glob': {},
                       'exps': [{} for e in self.exps],
                       'exp_keys': list(self.exp_keys),
                       'result_keys': list(self.result_keys),
                       'has_np': False,
                       }
            to_save_np = {}
            _split_for_save(self.glob, to_save['glob'], to_save_np, 'glob')
            for i, e in enumerate(self.exps):
                _split_for_save(e, to_save['exps'][i], to_save_np,
                                "exp_%d" % i)
            if len(to_save_np) > 0:
                to_save['has_np'] = True
                if compress:
                    np.savez_compressed(self.filename, **to_save_np)
                else:
                    np.savez(self.filename, **to_save_np)
            with open(self.filename + '.json', 'w') as f:
                json.dump(to_save, f, indent=2)

    @classmethod
    def load(cls, filename, set_filename=False):
        logger = Logger()
        with open(filename + '.json', 'r') as f:
            data = json.loads(f.read())
        logger.glob = data['glob']
        logger.exps = data['exps']
        logger.exp_keys = set(data['exp_keys'])
        logger.exp_results = set(data['result_keys'])
        if data['has_np']:
            data_np = np.load(filename + '.npz')
            for full_key in data_np.keys():
                p, k = full_key.split('_', 1)
                if p == 'glob':
                    logger.store_global(k, data_np[full_key])
                elif p == 'exp':
                    num, name = k.split('_', 1)
                    logger.exps[int(num)][name] = data_np[full_key]
        if set_filename:
            logger.filename = filename
        return logger


def _split_for_save(orig_dict, save_dict, save_dict_np, np_prefix):
    for k, v in orig_dict.items():
        if isinstance(v, np.ndarray):
            save_dict_np["%s_%s" % (np_prefix, k)] = np.array(v)
        else:
            save_dict[k] = v
