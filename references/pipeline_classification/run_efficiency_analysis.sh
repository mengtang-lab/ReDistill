##!/bin/bash

DIR=$0
STUDENT=$1
TEACHER=$2
DIST_CONFIG=$3
IM_SIZE=$4

PYTHONPATH="$(dirname $DIR)/":"/home/fangchen/WorkDir/Residual-Encoded-Distillation/":$PYTHONPATH \
python ./efficiency_analysis/efficiency_metrics_reed.py -s $STUDENT -t $TEACHER -d $DIST_CONFIG --im_size $IM_SIZE