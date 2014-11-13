#lfd
Base Repository for Learning from Demonstrations with Thin Plate Splines

##Installation
This code has been tested on Ubuntu 12.04.

###Dependencies
- [bulletsim](https://github.com/hojonathanho/bulletsim)
- [trajopt](http://rll.berkeley.edu/trajopt)
  - [OpenRAVE](http://openrave.org/docs/latest_stable/install) 0.9 or later
  - [PCL](http://www.pointclouds.org) 1.7
- Python 2.7
- SciPy >= 0.14
- NumPy >= 1.8.1
- [h5py](http://www.h5py.org)
- [joblib](http://packages.python.org/joblib)

###Instructions
- Install [bulletsim](https://github.com/hojonathanho/bulletsim) from source. Use the `lite` branch and follow its README instructions.
- Install [trajopt](http://rll.berkeley.edu/trajopt) from source. Follow the installation instructions but with the following modifications:
  - Use [this fork](https://github.com/erictzeng/trajopt) and the `trajopt-jointopt` branch instead.
  - Install [OpenRAVE](http://openrave.org/docs/latest_stable/install) 0.9
  - Install [PCL](http://www.pointclouds.org) 1.7. In Ubuntu, that is:
  ```
  sudo apt-get install libpcl-1.7-all
  ```
  - Run the `cmake` command with the option `BUILD_CLOUDPROC=ON`, that is:
  ```
  cmake /path/to/trajopt -DBUILD_CLOUDPROC=ON
  ```
- You can install SciPy, NumPy, h5py and joblib with pip install to get the latest versions.
  - Before installing h5py you may need to run
  ```
  sudo apt-get install libhdf5-dev
  ``` 

Add the following path to your PYTHONPATH:
```
/path/to/lfd
```

Now you should be able to run the scripts in the `examples` directory.


##Running the Test Suite
You can run the test suite using this command:
```
python -m unittest discover -s /path/to/lfd/test/
```

##Additional Installation to Use the GPU Functions

###Dependencies
- gfortran
- cmake
- boost-python
- [CUDA](https://developer.nvidia.com/cuda-downloads) >= 6.0
- PyCUDA >= 2013.1.1
- [CUDA scikit](http://scikit-cuda.readthedocs.org) >= 0.5.0
- [CULA](http://www.culatools.com/downloads/dense) >= R12 (optional)

###Instructions
- Install [CUDA](https://developer.nvidia.com/cuda-downloads) following [these instructions](http://www.r-tutor.com/gpu-computing/cuda-installation/cuda6.5-ubuntu).
- You can install PyCUDA with pip install to get the latest version.
- Install latest [CUDA scikit](http://scikit-cuda.readthedocs.org) from source (version available through pip doesn't have integration for the batched cublas calls yet):
```
git clone https://github.com/lebedov/scikits.cuda.git
cd scikits.cuda
sudo python setup.py install 
```
- CULA (optional):
  - Linear systems can optionally be solved on the GPU using the CULA Dense toolkit.
  - Download and install the full edition of [CULA](http://www.culatools.com/downloads/dense/). The full edition is required since the free edition only has single precision functions. The full edition is free for academic use, but requires registration.
  - As recommended by the installation, set the environment variables CULA_ROOT and CULA_INC_PATH to point to the CULA root and include directories. Also, add the CULA library directory to your LD_LIBRARY_PATH. On my linux development machine, that would be
  ```
  export CULA_ROOT=/usr/local/cula
  export CULA_INC_PATH=$CULA_ROOT/include
  export LD_LIBRARY_PATH=${CULA_ROOT}/lib64:$LD_LIBRARY_PATH
  ```
- Build the lfd sources with cmake as you would normally do:
```
mkdir build_lfd
cd build_lfd
cmake /path/to/lfd
make -j
```

To use the compiled libraries from python, add the following path to your PYTHONPATH:
```
/path/to/build_lfd/lib
```

For more information, check out the README from the [tpsopt](https://github.com/dhadfieldmenell/lfd/tree/dev/lfd/tpsopt) module.

##Miscellaneous
###Downloading test data
First navigate to the `bigdata` directory, and then run the `download.py` script.

###Cache files
By default, some functions cache results in the default cache directory `/path/to/lfd/.cache/`. If you are running out of space, consider deleting this directory.
