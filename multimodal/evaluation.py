"""Helpers to evaluate learning results."""


import scipy.sparse as sp
import numpy as np


def compare_labels_given_nb(reco_label_vect, true_label_vect):
    if len(reco_label_vect.shape) == 1:
        reco_label_vect = reco_label_vect[np.newaxis, :]
        true_label_vect = true_label_vect[np.newaxis, :]
    nb_ex = true_label_vect.shape[0]
    nb_true = true_label_vect.sum(axis=-1)
    # Sort -vect to get idx in decreasing order
    idx = np.argsort(-reco_label_vect, axis=-1)
    e = np.eye(true_label_vect.shape[-1])
    reco_given_nb = np.vstack([e[idx[i, :nb_true[i]], :].sum(axis=0)
                               for i in range(nb_ex)])
    return (reco_given_nb == true_label_vect).all(axis=-1)


def score_labels_given_nb(reco_label_vect, true_label_vect):
    return np.average(
            compare_labels_given_nb(reco_label_vect, true_label_vect))


def compare_labels_threshold(reco_label_vect, true_label_vect, threshold):
    return ((reco_label_vect >= threshold) == true_label_vect).all(axis=-1)


def score_labels_threshold(reco_label_vect, true_label_vect, threshold):
    return np.average(compare_labels_threshold(
                reco_label_vect, true_label_vect, threshold))


def chose_examples(labels, label_set=None, number=1):
    """Choses n example of each label.
    """
    if label_set is None:
        label_set = set(labels)
    out = []
    for l in label_set:
        start = -1
        for _ in range(number):
            start = labels.index(l, start + 1)
            out.append(start)
    return out


def evaluate_label_reco(reco_acti, true_labels):
    """Compare reconstructed label activations with true labels.
    """
    labels = np.asarray(true_labels)
    best_reco = reco_acti.argmax(axis=1)
    assert(best_reco.shape == labels.shape)
    # This can lead to wrong computation and easily happens when converting
    # from multiple label representation.
    return np.average(best_reco == labels)


# Deprecated
def scores_from_dists(dists, true_labels_0, true_labels_1=None, verbose=False):
    if true_labels_1 is None:
        assert(dists.shape[0] == dists.shape[1])
        true_labels_1 = true_labels_0
    matching = np.argmin(dists, axis=1)
    found_labels_0 = [true_labels_1[m] for m in matching]
    ok = [f == l for f, l in zip(found_labels_0, true_labels_0)]
    result = np.average(ok)
    if verbose:
        print(result)
    return result


def dists_to_found_labels(dists, ex_labels):
    matching = np.argmin(dists, axis=1)
    found_labels = [ex_labels[m] for m in matching]
    return found_labels


def found_labels_to_score(true, found):
    ok = [f == l for f, l in zip(found, true)]
    result = np.average(ok)
    return result


def found_labels_to_confusion(true, found, n_labels):
    """n_labels x n_labels matrix
    conf[i, j] is number of time label i has been classified as j.
    """
    conf = np.zeros((n_labels, n_labels))
    conf[true, found] += 1
    return conf


def todense(X):
    if sp.issparse(X):
        return np.asarray(X.todense())
    else:
        return X


def all_distances(reco_data, ex_data, measure):
    reco_data = todense(reco_data)[:, np.newaxis, :]
    ex_data = todense(ex_data)[np.newaxis, :, :]
    return measure(reco_data, ex_data, axis=-1)


def classify_NN(reco_data, ex_data, ex_labels, measure):
    """For each sample in reco_data, compares it with all examples of
    ex_data.

    test_data should not contain examples that appear in reco_data
    """
    dists = all_distances(reco_data, ex_data, measure)
    return dists_to_found_labels(dists, ex_labels)


def evaluate_NN_label(reco_data, test_data, true_labels, test_labels, measure):
    """For each sample in reco_data, compares it with all examples of
    test_data. The test is considered successful when the label corresponding
    to the sample (from true_labels) matches the label corresponding to the
    example (in test_labels).

    test_data should not contain examples that appear in reco_data
    """
    reco_data = todense(reco_data)[:, np.newaxis, :]
    test_data = todense(test_data)[np.newaxis, :, :]
    dists = measure(reco_data, test_data, axis=-1)
    return scores_from_dists(dists, true_labels, test_labels)
