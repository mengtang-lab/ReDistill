##!/bin/bash
ROOT="/home/fangchen/WorkDir/Residual-Encoded-Distillation/"


DIR=$0
DATASET=$1
TEACHER=$2
STUDENT=$3

if [[ $DATASET == imagenet ]]
then
DIST_METHOD=$4
DIST_CONFIG=$5
REED_CONFIG=$6
NUM_IMAGES=$7
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python efficiency_metrics_imagenet.py -t $TEACHER -s $STUDENT -d $DIST_METHOD \
                            --dist_config ../pipeline_classification/imagenet/configs/imagenet/$DIST_CONFIG \
                            --reed_config ../pipeline_classification/imagenet/configs/imagenet/$REED_CONFIG \
                            --im_size 224 --num_images $NUM_IMAGES
elif [[ $DATASET == stl10 ]]
then
DIST_CONFIG=$4
NUM_IMAGES=$5
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python efficiency_metrics_stl10.py -t $TEACHER -s $STUDENT \
                            -d ../base/configs/$DIST_CONFIG --im_size 128 --num_images $NUM_IMAGES
elif [[ $DATASET == ddpm ]]
then
DIST_CONFIG=$4
NUM_IMAGES=$5
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python efficiency_metrics_ddpm.py -t $TEACHER -s $STUDENT \
                            -d ../pipeline_diffusion/config/$DIST_CONFIG --im_size 32 --num_images $NUM_IMAGES
fi