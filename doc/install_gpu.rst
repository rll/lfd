.. _install_gpu:

Additional Installation to Use the GPU
======================================


Dependencies
------------

-  gfortran
-  cmake
-  boost-python
-  `CUDA <https://developer.nvidia.com/cuda-downloads>`_ >= 6.0
-  PyCUDA >= 2013.1.1
-  `CUDA scikit <http://scikit-cuda.readthedocs.org>`_ >= 0.5.0
-  `CULA <http://www.culatools.com/downloads/dense>`_ >= R12 (optional)


Instructions
------------

-  Install `CUDA <https://developer.nvidia.com/cuda-downloads>`_ following `these instructions <http://www.r-tutor.com/gpu-computing/cuda-installation/cuda6.5-ubuntu>`_.
-  You can install PyCUDA with pip install to get the latest version.
-  Install latest `CUDA scikit <http://scikit-cuda.readthedocs.org>`_ from source (version available through pip doesn't have integration for the batched cublas calls yet)::

      git clone https://github.com/lebedov/scikits.cuda.git
      cd scikits.cuda
      sudo python setup.py install

-  CULA (optional):
   
   -  Linear systems can optionally be solved on the GPU using the CULA Dense toolkit.
   -  Download and install the full edition of `CULA <http://www.culatools.com/downloads/dense/>`_. The full edition is required since the free edition only has single precision functions. The full edition is free for academic use, but requires registration.
   -  As recommended by the installation, set the environment variables ``CULA_ROOT`` and ``CULA_INC_PATH`` to point to the CULA root and include directories. Also, add the CULA library directory to your ``LD_LIBRARY_PATH``. On my linux development machine, that would be::
   
         export CULA_ROOT=/usr/local/cula
         export CULA_INC_PATH=$CULA_ROOT/include
         export LD_LIBRARY_PATH=${CULA_ROOT}/lib64:$LD_LIBRARY_PATH

-  Build the lfd sources with cmake as you would normally do::
   
      mkdir build_lfd
      cd build_lfd
      cmake /path/to/lfd
      make -j

To use the compiled libraries from python, add the following path to your ``PYTHONPATH``::
   
   /path/to/build_lfd/lib

For more information, check out the README from the `tpsopt <https://github.com/dhadfieldmenell/lfd/tree/dev/lfd/tpsopt>`_ module.
