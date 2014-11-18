.. _misc:

Miscellaneous
=============


Settings files
--------------
The ``lfd`` package and its subpackages have a ``settings.py`` file with setting variables. 
You can easily override these variables by creating a package ``lfd_settings`` with the same structure as ``lfd``. 
The variable that should be overriden can be defined in the corresponding ``settings.py`` file of the ``lfd_settings`` package.

This is best illustrated with an example.
Suppose you want to override the ``EM_ITER`` variable from the :mod:`lfd.registration.settings` module to be ``5`` instead.
First, you can generate the ``lfd_settings`` package with the provided script: ::

  cd /path/to/lfd
  python scripts/make_lfd_settings_package.py ../lfd_settings/lfd_settings

Remember to add the ``lfd_settings`` package to your ``PYTHONPATH``: ::

  export PYTHONPATH=/path/to/lfd_settings:$PYTHONPATH

Then, in the file ``/path/to/lfd_settings/lfd_settings/registration/settings.py``, add python code that overrides the ``EM_ITER`` variable: ::

  EM_ITER = 5


Downloading test data
---------------------

First navigate to the ``bigdata`` directory, and then run the ``download.py`` script.


Cache files
-----------
By default, some functions cache results in the default cache directory ``/path/to/lfd/.cache/``. If you are running out of space, consider deleting this directory.
