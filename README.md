Multimodal Learning with NMF
============================

*A set of tools and experimental scripts used to achieve multimodal learning with nonnegative matrix factorization (NMF).*

This repository contains code to reproduce the experiments from the publication:

O. Mangin, P.Y. Oudeyer, **Learning semantic components from sub symbolic multi modal perception** *Joint IEEE International Conference on Development and Learning and on Epigenetic Robotics (ICDL EpiRob)*, Osaka (Japan) (2013) ([More information][Mangin2013], [bibtex](http://olivier.mangin.com/media/bibtex/Mangin2013.bib))

Please consider citing this paper when re-using the code in scientific publications.


Usage
-----
This code is intended to work on its own. It however requires the dependencies mentioned below to be installed and available on the operating system. Please note the following elements.

- **Locations:** The file *local.py* is meant to hold local configuration of paths to data, features, etc. A template is provided in the repository. The paths defined in that file are only used when no path argument is provided to related functions (for examples the dataset loaders, or feature generators).
- **Data:** The experimental scripts require data (see below). This data must be accessible from the script and eventually pre-processed. For the motion data, pre-processing can be achieved through *multimodal/db/scripts/build_choreo2_features*. Alternatively, all features and metadata for the databases can be downloaded by running *multimodal/db/scripts/download_dbs*. This removes the need to pre-process the *choreo2* dataset.
- **Experiments:** Just run the experimental script. Be sure to generate the required features. For the moment, the ACORNS Caregiver dataset is not available online, however feature files can be provided on request.



Source files hierarchy:
-----------------------
###Directories
- **db:** helpers to load and pre-process data
- **features:** helpers to build features
- **lib:** general purpose helpers
- **experiments:** experimental scripts
- **test:** a few unit tests

###Root Files
- `experiment.py` This file contains the main logic of the experimental setup. It defines classes to run multimodal experiments in a consistent and simplified manner. These include data preparation, training and evaluation.
- `learner.py` Contains a class that abstracts the process of learning from multiple modalities. Contains the learning mechanism based on the NMF algorithm (itself included in `lib/nmf.py`).
- `evaluation.py` Helpers to evaluate learning results.
- `pairing.py` Helpers to generate associations of samples from different modalities with same labels.
- `plots.py` Plot functions to generate specific figures.

###Experimental scripts
- `experiments/icdl2013.py` Runs experiments and generate results corresponding to [Mangin2013][Mangin2013]. Uses *Acorns Caregiver* database for speech sounds as well as the *Choreo2* dataset.
- `experiments/image_sound.py` Runs experiments similar to [Mangin2013][Mangin2013] with speech sounds from the *Acorns Caregiver* database and images from the *image database*.
- `experiments/launcher.py` Script to run many experiments with variations of parameters. Requires [expjobs](http://github.com/omangin/expjobs) and some time or a cluster.
- `experiments/image_sound_eval_sliding_windows.py` Script to run experiments with speech sound and images in which the learner is evaluated on its recognition sound from small time windows.
- `experiments/plot_image_sound_eval_sliding_windows.py` Interactive and static plots from the previous experiment.
- `experiments/plot_info_matrix.py` Script to generate plot of the mutual information between sample labels and internal coefficients. The script re-uses a trained dictionary obtained from `launcher.py`.
- `experiments/two_modalities.py` and `experiments/three_modalities.py` run an experiment from a configuration file. Mainly used by `launcher.py`.

Data
----
- **Acorns Caregiver** is not available online. However, feature and meta files can be downloaded (see the *download_meta_and_features* function in *db/acorns.py* or use *multimodal/db/scripts/download_dbs*).
- **Choreo1** can be found at [flowers.inria.fr/choreo/doc](https://flowers.inria.fr/choreo/doc/index.html)
- **Choreo2** can be found at [flowers.inria.fr/choreo2](https://flowers.inria.fr/choreo2/index.html)
- **Object images** (not publicly available yet). Pictures acquired by [Natalia Lyubova and David Filliat][Lyubova2012] as frames from interaction with an iCub robot, through an RGBD sensor (red, green, and blue camera coupled with a depth sensor). Feature and meta files can be downloaded (see the *download_meta_and_features* function in *db/objects.py* or use *multimodal/db/scripts/download_dbs*).


License
-------
This code is distributed under the new BSD license. Please see LICENSE for more details.


Acknowledgement
---------------
- Uses *transformation.py* from [ROS tf package](http://wiki.ros.org/tf)
- Uses a few functions imported from [scikit-learn](http://scikit-learn.org). These are grouped in file *sklearn_utils.py*.
- Uses functions adapted from [prettyplotlib](http://olgabot.github.io/prettyplotlib/)


Plateform
---------

Requirements
------------
- python 2.7
- numpy
- scipy
- [librosa](http://github.com/bmcfee/librosa) (for sound feature computation)


[Mangin2013]: http://olivier.mangin.com/publi#Mangin.2013.ICDL
[Lyubova2012]: http://www.ensta-paristech.fr/TILDElyubova/data/ijcnn2012.pdf
