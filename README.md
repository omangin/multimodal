Multimodal Learning with NMF
============================

*A set of tools and experimental scripts used to achieve multimodal learning with nonnegative matrix factorization (NMF).*

This repository contains code to reproduce the experiments from the publication:

O. Mangin, P.Y. Oudeyer, **Learning semantic components from sub symbolic multi modal perception** *Joint IEEE International Conference on Development and Learning and on Epigenetic Robotics (ICDL EpiRob)*, Osaka (Japan) (2013) ([More information](http://olivier.mangin.com/publi#Mangin.2013.ICDL), [bibtex](http://olivier.mangin.com/media/bibtex/Mangin2013.bib))

Please consider citing this paper when re-using the code in scientific publications.


Usage
-----
This code is intended to work by its own. It however requires the dependencies mentioned below to be installed and available on the operating system. Please note the following elements.

- **Locations:** The file *local.py* is meant to hold local configuration of paths to data, features, etc. A template is provided in the repository. The paths defined in that file are only used when no path argument is provided to related functions (for examples the dataset loaders, or feature generators).
- **Data:** The experimental scripts require data (see below). This data must be accessible from the script and eventually pre-processed. For the motion data, pre-processing can be achieved through *multimodal/db/scripts/build_choreo2_features*.
- **Experiments:** Just run the experimental script. Be sure to generate the required features. For the moment, the ACORNS Caregiver dataset is not available online, however feature files can be provided on request.



Source files hierarchy:
-----------------------
- **db:** helpers to load and pre-process data
- **features:** helpers to build features
- **lib:** general purpose helpers
- **experiments:** experimental scripts
- **test:** a few unit tests

Data
----
- ACORNS Caregiver is not available online. However, I can send you feature files on request.
- Choreo1 can be found at [flowers.inria.fr/choreo/doc](https://flowers.inria.fr/choreo/doc/index.html)
- Choreo2 can be found at [flowers.inria.fr/choreo2](https://flowers.inria.fr/choreo2/index.html)


License
-------
This code is distributed under the new BSD license. Please see LICENSE for more details.


Acknowledgement
---------------
- Uses *transformation.py* from [ROS tf package](http://wiki.ros.org/tf)
- Uses a few functions imported from [scikit-learn](http://scikit-learn.org). These are grouped in file *sklearn_utils.py*.


Plateform
---------


Requirements
------------
- python 2.7
- numpy
- scipy
