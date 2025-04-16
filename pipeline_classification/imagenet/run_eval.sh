##!/bin/bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

DIR=$0
MODEL=$1
CKPT=$2
DATASET=$3

PYTHONPATH="$(dirname $DIR)/":"/home/fangchen/WorkDir/Residual-Encoded-Distillation/":$PYTHONPATH \
python tools/eval.py --model $MODEL --ckpt $CKPT --dataset $DATASET --batch-size 64