.. _documenting:

Documenting
===========


Dependencies
------------

-  `Sphinx <http://sphinx.pocoo.org>`_ >= 1.3b.


Instructions
------------

The documentation is generated from ReStructured Text using Sphinx. 

The documentation sources are in the ``doc/`` directory. To locally build the documentation, go to the ``doc/`` directory and run::

	make html

The built documentation will be in the ``_build/html/`` directory.

The online documentation can be found at `rll.berkeley.edu/lfd <http://rll.berkeley.edu/lfd>`_. Whenever new commits are pushed to the ``master`` branch, the docs are rebuilt from this branch (assuming the build doesn't fail).

Use `Google <http://google-styleguide.googlecode.com/svn/trunk/pyguide.html#Comments>`_ style docstrings for documenting code. These `Sections <http://sphinxcontrib-napoleon.readthedocs.org/en/latest/#sections>`_ can be used inside the docstrings. For docstring examples, see `Example Google Style Python Docstrings <http://sphinxcontrib-napoleon.readthedocs.org/en/latest/example_google.html#example-google-style-python-docstrings>`_ or the module :mod:`lfd.registration`.