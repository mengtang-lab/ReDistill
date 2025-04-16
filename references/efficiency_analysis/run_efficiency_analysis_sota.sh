##!/bin/bash
ROOT="/home/fangchen/WorkDir/Residual-Encoded-Distillation/"
MDISTILL_DIR="$ROOT/pipeline_classification/imagenet/"

DIR=$0
STUDENT=$1
TEACHER=$2
DIST_METHOD=$3
DIST_CONFIG=$4

PYTHONPATH="$(dirname $DIR)/":"$ROOT":"$MDISTILL_DIR":$PYTHONPATH \
python ./efficiency_metrics_sota.py -s $STUDENT -t $TEACHER -d $DIST_METHOD --dataset imagenet --dist_config $MDISTILL_DIR/configs/imagenet/$DIST_CONFIG