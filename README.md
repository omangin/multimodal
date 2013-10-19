Multimodal Learning with NMF
============================

This code was used in the publication:

O. Mangin, P.Y. Oudeyer, **Learning semantic components from sub symbolic multi modal perception** *Joint IEEE International Conference on Development and Learning and on Epigenetic Robotics (ICDL EpiRob)*, Osaka (Japan) (2013) ([More information](http://olivier.mangin.com/publi#Mangin.2013.ICDL), [bibtex](http://olivier.mangin.com/media/bibtex/Mangin2013.bib))

Please consider citing this paper when re-using the code in scientific publications.

Source files hierarchy:
-----------------------
- db: helpers to load and pre-process data
- features: helpers to build features
- lib: general purpose helpers
- samples: experimental scripts
- test: a few unit tests

Data
----
- ACORNS Caregiver
- Choreo1 can be found at [flowers.inria.fr/choreo/doc](https://flowers.inria.fr/choreo/doc/index.html)
- Choreo2 can be found at [flowers.inria.fr/choreo2](https://flowers.inria.fr/choreo2/index.html)

License
-------

Acknowledgement
---------------
- uses transformation.py from [ROS tf package](http://wiki.ros.org/tf)

Dependencies
------------
- python 2.7
- numpy
- scipy
- scikit-learn
