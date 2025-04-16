##!/bin/bash
ROOT="/home/fangchen/WorkDir/Residual-Encoded-Distillation/"
CONFIG_DIR="$ROOT/pipeline_classification/base/configs/"

DIR=$0
STUDENT=$1
TEACHER=$2
DIST_CONFIG=$3
IM_SIZE=$4

PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python ./efficiency_metrics_reed.py -s $STUDENT -t $TEACHER -d $CONFIG_DIR/$DIST_CONFIG --im_size $IM_SIZE