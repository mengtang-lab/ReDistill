##!/bin/bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1"
DEVICES=0

########### DATASET SETTINGS ############
DATASET=STL10
IMSIZE=128
BATCHSIZE=8

#DATASET=CIFAR100
#IMSIZE=32
#BATCHSIZE=128
#########################################

DIR=$0
MODE=$1
METHOD=$2

if [[ $MODE == single ]]
then
PYTHONPATH="$(dirname $DIR)/":"/home/fangchen/WorkDir/Residual-Encoded-Distillation/":$PYTHONPATH \
python train_single.py \
          --root ./ --seed 0 --devices $DEVICES \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --architecture $METHOD --pretrained '' \
          --epochs 300 --learning_rate 1e-2 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD \
          --log
elif [[ $MODE == distill ]]
then
TEACHER=$3
TEACHERPRETRAINED=$4
CONFIG=$5
PYTHONPATH="$(dirname $DIR)/":"/home/fangchen/WorkDir/Residual-Encoded-Distillation/":$PYTHONPATH \
python train_distill.py \
          --root ./ --seed 0 --devices $DEVICES \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --architecture $METHOD \
          --teacher $TEACHER --teacher_pretrained $TEACHERPRETRAINED \
          --dist_config $CONFIG --dist_pretrained '' \
          --epochs 300 --learning_rate 1e-2 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD \
          --log
fi
