#!/usr/bin/env python
import argparse
import os, subprocess, urllib2, tarfile

parser = argparse.ArgumentParser()
parser.add_argument("--rsync",action="store_true")
args = parser.parse_args()

assert os.getcwd().endswith("bigdata")

if args.rsync: 
    subprocess.check_call("rsync -azvu pabbeel@rll.berkeley.edu:/var/www/lfd/bigdata/ ./ --exclude '*.py'", shell=True)
else:
    print "downloading tar file (this might take a while)"
    urlinfo = urllib2.urlopen("http://rll.berkeley.edu/lfd/bigdata/all.tar.gz")
    with open("all.tar.gz","w") as fh:
        fh.write(urlinfo.read())

print "unpacking file"
with tarfile.open("all.tar.gz") as tar:
    tar.extractall(".")
