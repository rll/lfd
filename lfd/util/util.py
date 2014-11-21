from __future__ import division

import os 
import time
import argparse
from lfd.util import colorize

def redprint(msg):
    print colorize.colorize(msg, "red", bold=True)

def yellowprint(msg):
    print colorize.colorize(msg, "yellow", bold=True)

class ArgumentParser(argparse.ArgumentParser):
    def parse_args(self, *args, **kw):
        res = argparse.ArgumentParser.parse_args(self, *args, **kw)
        from argparse import _HelpAction, _SubParsersAction
        for x in self._subparsers._actions:
            if not isinstance(x, _SubParsersAction):
                continue
            v = x.choices[res.subparser_name] # select the subparser name
            subparseargs = {}
            for x1 in v._optionals._actions: # loop over the actions
                if isinstance(x1, _HelpAction): # skip help
                    continue
                n = x1.dest
                if hasattr(res, n): # pop the argument
                    subparseargs[n] = getattr(res, n)
                    delattr(res, n)
            res.__setattr__(res.subparser_name, argparse.Namespace(**subparseargs))
        return res

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

# Define a context manager to suppress stdout
class suppress_stdout(object):
    '''
    A context manager for doing a "deep suppression" of stdout in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
    '''
    def __init__(self):
        # Open a null file
        while (True):
            try:
                self.null_fds =  os.open(os.devnull,os.O_RDWR)
                break
            except OSError:
                time.sleep(1)
        # Save the actual stdout file descriptor
        self.save_fds = os.dup(1)

    def __enter__(self):
        # Assign the null pointers to stdout
        os.dup2(self.null_fds,1)
        os.close(self.null_fds)

    def __exit__(self, *_):
        # Re-assign the real stdout back
        os.dup2(self.save_fds,1)
        # Close the null file
        os.close(self.save_fds)
