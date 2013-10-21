import random

from .db.acorns import load as load_acorns
from .db.choreo2 import load as load_choreo


def named_labels_to_range(sample_labels, shuffle=False):
    names = list(set(sample_labels))
    if shuffle:
        random.shuffle(names)
    inversion = dict((n, i) for i, n in enumerate(names))
    return names, [inversion[n] for n in sample_labels]


def organize_by_values(l, nb_values):
    buckets = [[] for i in range(nb_values)]
    for i, x in enumerate(l):
        buckets[x].append(i)
    return buckets


def associate(l):
    # buckets_tuple: one bucket for each modality grouped in a tuple
    return [zip(*buckets_tuple) for buckets_tuple in zip(*l)]


def flatten(l):
    return sum(l, [])


def associate_labels(labels_by_modality, shuffle=False):
    """ labels_by_modality: for each modality, list of labels for each sample.
        suffle: shuffle label associations (the association between samples is
                always shuffled).
    """
    # Replace label names by index in range
    names, new_labels = zip(*[named_labels_to_range(l)
                              for l in labels_by_modality])
    # Ensure all modalities have same number of labels
    n_labels = len(names[0])
    assert(all([len(l) == n_labels for l in names]))
    # Organize labels
    by_labels = [organize_by_values(l, n_labels) for l in new_labels]
    # Shuffle associations
    for l_mod in by_labels:
        for l in l_mod:
            random.shuffle(l)
    # Associate
    associated = associate(by_labels)
    new_labels = [[i] * len(l) for i, l in enumerate(associated)]
    return names, flatten(new_labels), flatten(associated)


def get_sound_labels(speaker):
    db = load_acorns(1, blacklist=True)
    return [db.tags[r.tags[0]] for r in db.records[speaker]]


def get_motion_labels():
    motion_db = load_choreo(verbose=False)
    motion_names = motion_db.label_descriptions
    return [motion_names[r[1][0]] for r in motion_db.records]


def associate_sound_motion(speaker):
    return associate_labels([get_sound_labels(speaker), get_motion_labels()])
