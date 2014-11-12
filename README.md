#lfd
Base Repository for Learning from Demonstrations with Thin Plate Splines

##Installation
This code has been tested on Ubuntu 12.04.

###Dependencies
- bulletsim
- trajopt
- h5py
- joblib
- CULA R12 or later (recommended)

###Instructions
- Install [bulletsim](https://github.com/hojonathanho/bulletsim) from source. Use the `lite` branch and follow its README instructions.
- Install [trajopt](http://rll.berkeley.edu/trajopt) from source. Follow the installation instructions except that use [this fork](https://github.com/erictzeng/trajopt) and the `trajopt-jointopt` branch.
- Install h5py. You can install it using `pip`:
```
sudo pip install h5py
```
- Install joblib. You can install it using `easy_install`:
```
sudo easy_install joblib
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

Now you should be able to run the scripts in the `examples` directory.

You can run the test suite using this command:
```
python -m unittest discover -s /path/to/lfd/test/
```

##Miscellaneous
###Downloading test data
First navigate to the `bigdata` directory, and then run the `download.py` script.

###Cache files
By default, some functions cache results in the default cache directory `/path/to/lfd/.cache/`. If you are running out of space, consider deleting this directory.
