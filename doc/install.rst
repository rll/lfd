.. _install:

Installation
============

This code has been tested on Ubuntu 12.04.


Dependencies
------------

-  `bulletsim <https://github.com/hojonathanho/bulletsim>`_
-  `trajopt <http://rll.berkeley.edu/trajopt>`_
   
   -  `OpenRAVE <http://openrave.org/docs/latest_stable/install>`_ >= 0.9
   -  `PCL <http://www.pointclouds.org>`_ 1.7
-  Python 2.7
-  NumPy >= 1.8.1
-  SciPy >= 0.9
-  HDF5
-  `h5py <http://www.h5py.org>`_
-  `joblib <http://packages.python.org/joblib>`_


Instructions
------------

-  Install `bulletsim <https://github.com/hojonathanho/bulletsim>`_ from source. Use the ``lite`` branch and follow its README instructions.
-  Install `trajopt <http://rll.berkeley.edu/trajopt>`_ from source. Follow the installation instructions but with the following modifications:
   
   -  Use `this fork <https://github.com/erictzeng/trajopt>`_ and the ``trajopt-jointopt`` branch instead.
   -  Install `OpenRAVE <http://openrave.org/docs/latest_stable/install>`_ 0.9 or later from the `OpenRAVE testing <https://launchpad.net/~openrave/+archive/testing>`_ PPA. In Ubuntu, that is::
      
         sudo add-apt-repository ppa:openrave/testing
         sudo apt-get update
         sudo apt-get install openrave

   -  Install `PCL <http://www.pointclouds.org>`_ 1.7. In Ubuntu, that is::
      
         sudo apt-get install libpcl-1.7-all

   -  Run the ``cmake`` command with the option ``BUILD_CLOUDPROC=ON``, that is::
      
         cmake /path/to/trajopt -DBUILD_CLOUDPROC=ON

-  Install NumPy, SciPy and HDF5. In Ubuntu, that is::
   
      sudo apt-get install python-numpy python-scipy libhdf5-serial-dev

- Install h5py and joblib with pip::
   
      sudo pip install h5py
      sudo pip install joblib


Add the following path to your ``PYTHONPATH``::

   /path/to/lfd

Now you should be able to run the scripts in the ``examples`` directory.


Running the Test Suite
----------------------

You can run the test suite using this command::

   python -m unittest discover -s /path/to/lfd/test/
