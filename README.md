#lfd
Base Repository for Learning from Demonstrations with Thin Plate Splines

##Installation
This code has been tested on Ubuntu 12.04.

###Dependencies
- bulletsim
- trajopt
- h5py
- joblib

###Instructions
- Install [bulletsim ](https://github.com/hojonathanho/bulletsim) from source. Use the `lite` branch and follow its README instructions.
- Install [trajopt](http://rll.berkeley.edu/trajopt) from source. Follow the installation instructions except that use [this fork](https://github.com/erictzeng/trajopt) and the `trajopt-jointopt` branch.
- Install h5py. You can install it using `pip`:
```
sudo apt-get install libhdf5-dev
sudo pip install h5py
```
- Install joblib. You can install it using `easy_install`:
```
sudo easy_install joblib
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
