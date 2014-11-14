#!/bin/sh

index=_build/html/index.html
if [ `uname` = Linux ] 
then
    google-chrome $index
else 
    open -a Google\ Chrome  $index
fi