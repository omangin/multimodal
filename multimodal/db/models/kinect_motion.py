# encoding: utf-8


__author__ = 'Olivier Mangin <olivier.mangin@inria.fr>'
__date__ = '04/2012'


"""Tools to access motion from a motion database.
"""


import os
import json
from collections import OrderedDict

import numpy as np
from scipy.io import savemat


DATA_DIR_SUFFIX = '_data'


class MotionDatabase:
    """Motion database.

    Parameters
    ----------

    name: string
        Name of the database.

    marker_names:
        List of names of the recorder markers. The order corresponds to
        recorded values.

    label_descriptions:
        Default to None.

    Attributes
    ----------

    records: (data, list of tags)
        Records are couples of a numpy data array and a list of tags or labels.

        Data arrays are of shape (T, D * nb_markers), where T is the length of
        the record, nb_markers the total number of markers, and D the dimension
        of each sample. In the case where samples are frames, there are three
        values to describe the frame translation and four for its rotation,
        which corresponds to D = 7.

    label_descriptions: Mapping
        Descriptions attached to labels. Either None or a mapping from labels
        to strings.
    """

    def __init__(self, name, marker_names, label_descriptions=None):
        self.name = name
        self.marker_names = marker_names
        self.records = []
        self.label_descriptions = label_descriptions

    def has_label_descriptions(self):
        return self.label_descriptions is not None

    def get_description_safe(self, label):
        if self.has_label_descriptions():
            try:
                return self.label_descriptions[label]
            except (IndexError, KeyError):
                pass
        return ''

    def get_data_dimension(self):
        if len(self.records) == 0:
            raise ValueError(
                    'Empty database, can\'t figure data dimension.')
        else:
            return self.records[0][0].shape[1:]

    def data(self):
        """Iterator through data.
        """
        for (d, l) in self.records:
            yield d

    def labels(self):
        """Iterator through labels.
        """
        for (d, l) in self.records:
            yield l

    def add(self, data, label):
        """adds data (as array) and label to database.
        """
        self.records.append((data, label))

    def size(self):
        return len(self.records)

    def get_occuring_labels(self):
        """Returns the set of labels from records.
        """
        return set([l for d, ls in self.records for l in ls])

    def extend(self, other):
        """Extends a database with records from an other.
        """
        assert isinstance(other, MotionDatabase)
        if self.marker_names != other.marker_names:
            raise ValueError('Trying to merge two databases containing '
                    + 'different marker names.')
        # Check that label descriptions coincidate on occuring labels
        labels = self.get_occuring_labels().union(other.get_occuring_labels())
        if not all([self.label_descriptions[l] == other.label_descriptions[l]
            for l in labels]):
            raise ValueError('Different descriptions of occuring labels.')
        self.records.extend(other.records)

    def export(self, destination, filename=None, data_type='npz',
            force_flat=False):
        """Export database as a meta data json file and a data file.

        Parameters
        ----------

        destination: string
            Location where to export database.

        filename: string
            Alternative name for output data.
            Default: name attribute.

        data_type: {'npz', 'txt', 'mat'}
            Default: 'npz'

        force_flat: boolean, default: False
            For txt export only, flatten the extra dimensions if the
            array has more than two dimensions.

        """
        if filename is None:
            filename = self.name

        meta = self._get_meta(filename, data_type=data_type)

        if data_type == 'npz':
            # Save meta-data
            with open(os.path.join(destination, filename + '.json'), 'w') as f:
                    json.dump(meta, f, indent=2)
            # Save data
            self.save_data_npz(os.path.join(destination, meta['data_file']))

        elif data_type == 'txt':
            # Prepare destination
            dest_dir, data_dir = MotionDatabase.create_text_subdirs(
                    destination, filename)
            # Save meta-data
            with open(os.path.join(dest_dir, filename + '.json'), 'w') as f:
                    json.dump(meta, f, indent=2)
            # Save data
            self.save_data_txt(data_dir, force_flat=force_flat)

        elif data_type == 'mat':
            savemat(os.path.join(destination, filename + '.mat'), meta)

        else:
            raise ValueError("Wrong data type %s" % data_type)

    def _get_meta(self, filename, data_type='npz'):
        meta = OrderedDict()
        meta['name'] = self.name
        meta['marker_names'] = self.marker_names
        if self.has_label_descriptions() or not data_type == 'mat':
                meta['label_descriptions'] = self.label_descriptions
        meta_records = [
                    {'data_id': i, 'labels': l}
                    for (i, l) in enumerate(self.labels())
                    ]
        if data_type == 'npz':
            meta['data_file'] = filename + '.' + data_type

        elif data_type == 'txt':
            meta['data_dir'] = filename + DATA_DIR_SUFFIX

        if data_type in ['npz', 'txt']:
            meta['records'] = meta_records

        elif data_type == 'mat':
            meta['data'] = [d for d in self.data()]
            meta['labels'] = [l for l in self.labels()]

        else:
            raise ValueError("Wrong data type %s" % data_type)

        return meta

    def save_data_npz(self, dest_file):
        # Use id as name for each data array inside npz archive
        np.savez(dest_file,
                **dict([(str(i), d[0]) for i, d in enumerate(self.records)])
                )

    def save_data_txt(self, data_dir, force_flat=False):
        for i, d in enumerate(self.data()):
            # Each example is stored in separate file
            # 2D arrays are used
            ad = np.array(d)
            if len(ad.shape) != 2:
                if force_flat and len(ad.shape) > 2:
                    s = ad.shape
                    new_shape = (s[0], np.prod(s[1:]))
                    ad = ad.reshape(new_shape)
                else:
                    raise ValueError(
                            "Got shape %s. only 2D array can be exported"
                            % str(ad.shape)
                            + "to text. See 'force_flat' argument to "
                            + "overcome."
                            )
            np.savetxt(os.path.join(data_dir, str(i) + '.txt'), ad)

    def print_info(self):
        print("Each data example is a (%s) array." % ', '.join(
            ['T'] + [str(i) for i in self.get_data_dimension()]
            ))
        print("The second dimension corresponds to markers:")
        print("\t- %s" % '\n\t- '.join(self.marker_names))

    @classmethod
    def load_from_npz(cls, location, verbose=False):
        """Load database from meta file.

        Parameters
        ----------

        location: string or path
            Location of the json meta file.

        Returns
        -------

        Database object.

        """
        with open(location, 'r') as meta_file:
            meta = json.load(meta_file)
            # meta is a dictionary containg data from the json file
        path_to_data = os.path.join(os.path.dirname(location),
                meta['data_file'])
        loaded_data = np.load(path_to_data)
        db = MotionDatabase(meta['name'], meta['marker_names'],
                label_descriptions=meta['label_descriptions'])
        for r in meta['records']:
            d = loaded_data[str(r['data_id'])]  # numpy array
            l = r['labels']  # list of labels as integers
            db.records.append((d, l))
        if verbose:
            print("Loaded %d examples for ``%s`` set." %
                  (len(db.records), meta['name']))
            db.print_info()
        return db

    @classmethod
    def create_text_subdirs(cls, destination, filename):
        """Creates (if they don't exist) and return base and data directories
        for text db.
        """
        dir_name = os.path.join(destination, filename)
        data_dir = os.path.join(dir_name, filename + DATA_DIR_SUFFIX)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        return (dir_name, data_dir)
