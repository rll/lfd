import sys, os

def suppress_stdout():
    devnull = open('/dev/null', 'w')
    oldstdout_fno = os.dup(sys.stdout.fileno())
    oldstderr_fno = os.dup(sys.stderr.fileno())
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    return (oldstdout_fno, oldstderr_fno)

def unsuppress_stdout(oldstdout_fno, oldstderr_fno):
    os.dup2(oldstdout_fno, 1)
    os.dup2(oldstderr_fno, 2)

