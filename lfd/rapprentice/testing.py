import sys, traceback
from time import time
from rapprentice.colorize import colorize

TEST_FUNCS = {}


def testme(func):    
    TEST_FUNCS[func.__name__] = func
    return func
def test_all(stop=False):
    nPass,nFail = 0,0
    for (name,func) in TEST_FUNCS.items():
        print colorize("function: %s"%name,"green")
        try:
            t_start = time()
            func()
            t_elapsed = time() - t_start
            print colorize("PASSED (%.3f sec)"%t_elapsed,"blue")
            nPass += 1
        except Exception:    
            traceback.print_exc(file=sys.stdout)
            if stop: raise
            print colorize("FAILED","red")
            nFail += 1
            
            
    print "%i passed, %i failed"%(nPass,nFail)