import random

from .lib.window import concat_of_frames


def named_labels_to_range(sample_labels, shuffle=False):
    names = list(set(sample_labels))
    if shuffle:
        random.shuffle(names)
    inversion = dict((n, i) for i, n in enumerate(names))
    return names, [inversion[n] for n in sample_labels]


def associate_labels(labels_by_modality, shuffle=False):
    return zip(*[named_labels_to_range(l, shuffle=shuffle)
                 for l in labels_by_modality])


def organize_by_values(l, nb_values=None, indices=None):
    if nb_values is None:
        nb_values = len(set(l))
    buckets = [[] for i in range(nb_values)]
    enum = enumerate(l) if indices is None else zip(indices, l)
    for i, x in enum:
        buckets[x].append(i)
    return buckets


def associate(l):
    # buckets_tuple: one bucket for each modality grouped in a tuple
    return [zip(*buckets_tuple) for buckets_tuple in zip(*l)]


def flatten(l):
    return sum(l, [])


def associate_samples(labels_by_modality, shuffle=False):
    """ labels_by_modality: for each modality, list of labels for each sample.
        suffle: shuffle label associations (the association between samples is
                always shuffled).
    """
    # Replace label names by index in range
    names, new_labels = associate_labels(labels_by_modality, shuffle=shuffle)
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


def associate_to_window(labels_window, frame_index, frame_labels, frame_rate):
    """Create ArrayWindow containing frame indices so that at sample reference
    times, the label of the frame matches the one from the labels_window.

    Labels are supposed to be matched between the modalities.

    Illustration (brackets indicates the labels):
    |           Window [1]          |       Second Window [2]           |
    | Frame [1] | Frame [1] | Frame [1] | Frame [2] | Frame [2] | Frame [2] !
     <--- T ---> <--- T ---> ...
                                     <-> => exceeded time for frames of win 1
    T: frame period (1. / frame_rate)
    """
    # Ensure all modalities have same number of labels
    grouped_frame_by_labels = organize_by_values(frame_labels,
                                                 indices=frame_index)
    for l in grouped_frame_by_labels:
        random.shuffle(l)
    framed_window = concat_of_frames(labels_window.absolute_start,
                                     labels_window.absolute_end, frame_rate)
    for f in framed_window.windows:
        # Get label for current time
        label = labels_window.get_subwindow_at(f.absolute_start).obj
        try:
            # Get an index for the current label
            f.obj = grouped_frame_by_labels[label].pop()
        except IndexError:
            raise ValueError('Not enough frames to match labels '
                             '(for label {}).'.format(label))
    return framed_window
