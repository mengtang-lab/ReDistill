##!/bin/bash

DIR=$0
STUDENT=$1
TEACHER=$2
DIST_METHOD=$3
DATASET=$4
DIST_CONFIG=$5

PYTHONPATH="$(dirname $DIR)/":"/home/fangchen/WorkDir/Residual-Encoded-Distillation/":"$(dirname $DIR)/other_distillation_methods/MultiLevelLogitDistillation/":$PYTHONPATH \
python ./efficiency_analysis/efficiency_metrics_sota.py -s $STUDENT -t $TEACHER -d $DIST_METHOD --dataset $DATASET --dist_config $DIST_CONFIG