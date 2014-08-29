#!/usr/bin/env python

import argparse
import os, subprocess, tarfile

parser = argparse.ArgumentParser()
parser.add_argument("--delete", action="store_true")
args = parser.parse_args()

assert os.getcwd().endswith("bigdata")

print "creating tar file"
with tarfile.open("all.tar.gz", "w") as tar:
    for fname in os.listdir("."):
        if not (fname.endswith("py") or fname.endswith("tar.gz")):
            tar.add(fname)

print "uploading"
subprocess.check_call("rsync -azvu %s all.tar.gz pabbeel@rll.berkeley.edu:/var/www/lfd/bigdata/ --exclude '*.py'"%("--delete" if args.delete else ""), shell=True)
