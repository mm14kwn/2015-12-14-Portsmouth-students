#!/bin/bash

source activate py27
NB_FILES=`find ~/swc_repo -name '*.ipynb'`
echo $NB_FILES
for i in $NB_FILES
do
    ipython nbconvert --to latex $i
done

TEX_FILES=`find ~/swc_repo -name '*.tex'`
for i in $TEX_FILES
do
    cp $i ~/python/swc_pdfs/
done
