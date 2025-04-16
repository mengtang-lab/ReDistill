#!/bin/bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1"

DIR=$0
MODE=$1
MODEL=$2

if [[ $MODE == single ]]
then

PYTHONPATH="$(dirname $DIR)/":"/home/fangchen/WorkDir/Residual-Encoded-Distillation/":$PYTHONPATH \
torchrun --nproc_per_node=2 train_single.py \
     --data-path /mnt/data/imagenet2012/ \
     --model $MODEL \
     --epochs 300 --lr 0.045 --wd 0.00004 \
     --lr-step-size 1 --lr-gamma 0.98 \
     --output-dir "./output/$MODEL-single/"

elif [[ $MODE == distill ]]
then

TEACHERS=$3
DIST_CONFIG=$4

PYTHONPATH="$(dirname $DIR)/":"/home/fangchen/WorkDir/Residual-Encoded-Distillation/":$PYTHONPATH \
torchrun --nproc_per_node=2 train_distillation.py \
     --data-path /mnt/data/imagenet2012/ \
     --teachers $TEACHERS --student $MODEL \
     --dist_config $DIST_CONFIG --dist_rate 0.5 \
     --epochs 300 --lr 0.045 --wd 0.00004 \
     --lr-step-size 1 --lr-gamma 0.98 \
     --output-dir "./output/S=$MODEL,T=[$TEACHERS]/"
fi