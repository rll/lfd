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
-  `CUDA SciKit <http://scikit-cuda.readthedocs.org>`_ >= 0.5.0
-  Mako
-  `CULA <http://www.culatools.com/downloads/dense>`_ >= R12 (optional)


Instructions
------------

-  CUDA:

   -  Get the CUDA installers from the `CUDA download site <https://developer.nvidia.com/cuda-downloads>`_ and install it. ::

         sudo dpkg -i cuda-repo-ubuntu1204_6.5-14_amd64.deb
         sudo apt-get update

   -  Then you can install the CUDA Toolkit using apt-get. ::
   
         sudo apt-get install cuda

   -  You should reboot the system afterwards and verify the driver installation with the nvidia-settings utility.
   -  Set the environment variable ``CUDA_HOME`` to point to the CUDA home directory. Also, add the CUDA binary and library directory to your ``PATH`` and ``LD_LIBRARY_PATH``. ::
   
         export CUDA_HOME=/usr/local/cuda
         export PATH=${CUDA_HOME}/bin:${PATH}   
         export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

-  Install PyCUDA with pip. Make sure that ``PATH`` is defined as root. ::

      sudo PATH=$PATH pip install pycuda

-  Install CUDA SciKit with pip. ::

      sudo pip install pycuda scikits.cuda>=0.5.0a1 Mako

-  CULA (optional):
   
   -  Linear systems can optionally be solved on the GPU using the CULA Dense toolkit.
   -  Download and install the full edition of `CULA <http://www.culatools.com/downloads/dense/>`_. The full edition is required since the free edition only has single precision functions. The full edition is free for academic use, but requires registration.
   -  As recommended by the installation, set the environment variables ``CULA_ROOT`` and ``CULA_INC_PATH`` to point to the CULA root and include directories. Also, add the CULA library directory to your ``LD_LIBRARY_PATH``. ::
   
         export CULA_ROOT=/usr/local/cula
         export CULA_INC_PATH=$CULA_ROOT/include
         export LD_LIBRARY_PATH=${CULA_ROOT}/lib64:$LD_LIBRARY_PATH

-  Build the lfd sources with cmake as you would normally do. ::
   
      mkdir build_lfd
      cd build_lfd
      cmake /path/to/lfd
      make -j

To use the compiled libraries from python, add the following path to your ``PYTHONPATH``: ::
   
   /path/to/build_lfd/lib

For more information, check out the README from the `tpsopt <https://github.com/rll/lfd/tree/master/lfd/tpsopt>`_ module.
