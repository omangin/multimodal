Multimodal Learning with NMF
============================

*A set of tools and experimental scripts used to achieve multimodal
learning with nonnegative matrix factorization (NMF).*

This repository contains code to reproduce the experiments from the
publications:

- O. Mangin, D. Filliat, L. ten Bosch, P.Y. Oudeyer, **MCA-NMF: Multimodal concept acquisition with non-negative matrix factorization** *PLOS ONE* (October 21, 2015) (more information: `[Mangin2015]`_, `bibtex <http://olivier.mangin.com/media/bibtex/Mangin2015.bib>`__)

- O. Mangin, P.Y. Oudeyer, **Learning semantic components from sub symbolic multi modal perception** *Joint IEEE International Conference on Development and Learning and on Epigenetic Robotics (ICDL EpiRob)*, Osaka (Japan) (2013) (more information: `[Mangin2013]`_, `bibtex <http://olivier.mangin.com/media/bibtex/Mangin2013.bib>`__)

Please consider citing these papers when re-using the code in scientific publications.

Usage
-----

This code is intended to work on its own. It however requires the
dependencies mentioned below to be installed and available on the
operating system. Please note the following elements.

-  **Locations:** The file *local.py* is meant to hold local
   configuration of paths to data, features, etc. A template is provided
   in the repository. The paths defined in that file are only used when
   no path argument is provided to related functions (for examples the
   dataset loaders, or feature generators).
-  **Data:** The experimental scripts require data (see below). This
   data must be accessible from the script and eventually pre-processed.
   For the motion data, pre-processing can be achieved through
   ``multimodal/db/scripts/build_choreo2_features``. Alternatively, all
   features and metadata for the databases can be downloaded by running
   ``multimodal/db/scripts/download_dbs``. This removes the need to
   pre-process the *choreo2* dataset.
-  **Experiments:** Just run the experimental script. Be sure to
   generate the required features. For the moment, the ACORNS Caregiver
   dataset is not available online, however feature files can be
   provided on request.

So for a quick hand on, just:

1. Install dependencies:

  .. code:: sh

    pip install numpy scipy matplotlib librosa

2. Download and install the sources:

  .. code:: sh

    git clone http://github.com/omangin/multimodal.git
    cd multimodal
    python setup.py develop --user

3. Download databases features and metadata (run from where you downloaded the repository):

  .. code:: sh

    ./multimodal/db/scripts/download_dbs

4. Reproduce `[Mangin2013]`_'s experiment:

  .. code:: sh

    ./multimodal/experiments/icdl2013.py

Source files hierarchy:
-----------------------

Directories
~~~~~~~~~~~

-  **db:** helpers to load and pre-process data
-  **features:** helpers to build features
-  **lib:** general purpose helpers
-  **experiments:** experimental scripts
-  **test:** a few unit tests

Main files in ``multimodal``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``experiment.py`` This file contains the main logic of the
   experimental setup. It defines classes to run multimodal experiments
   in a consistent and simplified manner. These include data
   preparation, training and evaluation.
-  ``learner.py`` Contains a class that abstracts the process of
   learning from multiple modalities. Contains the learning mechanism
   based on the NMF algorithm (itself included in ``lib/nmf.py``).
-  ``evaluation.py`` Helpers to evaluate learning results.
-  ``pairing.py`` Helpers to generate associations of samples from
   different modalities with same labels.
-  ``plots.py`` Plot functions to generate specific figures.

Experimental scripts (in ``samples``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``icdl2013.py`` Runs experiments and generate results
   corresponding to
   `Mangin2013 <http://olivier.mangin.com/publi#Mangin.2013.ICDL>`__.
   Uses *Acorns Caregiver* database for speech sounds as well as the
   *Choreo2* dataset.
-  ``image_sound.py`` Runs experiments similar to
   `Mangin2013 <http://olivier.mangin.com/publi#Mangin.2013.ICDL>`__
   with speech sounds from the *Acorns Caregiver* database and images
   from the *image database*.
-  ``launcher.py`` Script to run many experiments with
   variations of parameters. Requires
   `expjobs <http://github.com/omangin/expjobs>`__ and some time or a
   cluster.
-  ``image_sound_eval_sliding_windows.py`` Script to run
   experiments with speech sound and images in which the learner is
   evaluated on its recognition sound from small time windows.
-  ``plot_image_sound_eval_sliding_windows.py`` Interactive
   and static plots from the previous experiment.
-  ``plot_info_matrix.py`` Script to generate plot of the
   mutual information between sample labels and internal coefficients.
   The script re-uses a trained dictionary obtained from
   ``launcher.py``.
-  ``two_modalities.py`` and
   ``three_modalities.py`` run an experiment from a
   configuration file. Mainly used by ``launcher.py``.

Data
----

-  **Acorns Caregiver** is available
   `online <https://corpus1.mpi.nl/ds/asv/;jsessionid=0717131F4474EDF6A9002460E8921321?0&openhandle=hdl:1839/00-0000-0000-001A-D60B-1>`__.
   Please refer to `the article by Bergmann et
   al. <http://dx.doi.org/10.1371/journal.pone.0132245>`__ for
   permission. However, feature and metadata files can be downloaded
   through the code (see the ``download_meta_and_features`` function in
   ``db/acorns.py`` or use ``multimodal/db/scripts/download_dbs``). See
   also |DOI-Acorns|.
-  **Choreo1** can be found at
   `flowers.inria.fr/choreo/doc <https://flowers.inria.fr/choreo/doc/index.html>`__.
-  **Choreo2** can be found at
   `flowers.inria.fr/choreo2 <https://flowers.inria.fr/choreo2/index.html>`__.
   See also |DOI-Choreo| for features.
-  **Object images** (not publicly available yet). Pictures acquired by
   `Natalia Lyubova and David
   Filliat <http://www.ensta-paristech.fr/TILDElyubova/data/ijcnn2012.pdf>`__
   as frames from interaction with an iCub robot, through an RGBD sensor
   (red, green, and blue camera coupled with a depth sensor). Feature
   and meta files can be downloaded (see the
   ``download_meta_and_features`` function in ``db/objects.py`` or use
   ``multimodal/db/scripts/download_dbs``). See also |DOI-Images|.

License
-------

This code is distributed under the new BSD license. Please see LICENSE
for more details.

Acknowledgement
---------------

-  Uses ``transformation.py`` from `ROS tf
   package <http://wiki.ros.org/tf>`__.
-  Uses a few functions imported from
   `scikit-learn <http://scikit-learn.org>`__. These are grouped in file
   ``sklearn_utils.py``.
-  Uses functions adapted from
   `prettyplotlib <http://olgabot.github.io/prettyplotlib/>`__.

Requirements
------------

-  python >2.7 or >3
-  numpy
-  scipy
-  `librosa <http://github.com/bmcfee/librosa>`__ (for sound feature
   computation)

.. _[Mangin2013]: http://olivier.mangin.com/publi#Mangin.2013.ICDL
.. _[Mangin2015]: http://olivier.mangin.com/publi#Mangin.2015.PONE
.. |DOI-Acorns| image:: https://zenodo.org/badge/doi/10.5281/zenodo.29600.svg
   :target: http://dx.doi.org/10.5281/zenodo.29600
.. |DOI-Choreo| image:: https://zenodo.org/badge/doi/10.5281/zenodo.29602.svg
   :target: http://dx.doi.org/10.5281/zenodo.29602
.. |DOI-Images| image:: https://zenodo.org/badge/doi/10.5281/zenodo.29607.svg
   :target: http://dx.doi.org/10.5281/zenodo.29607
