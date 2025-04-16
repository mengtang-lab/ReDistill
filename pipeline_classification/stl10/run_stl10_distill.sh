##!/bin/bash
ROOT="/home/fangchen/WorkDir/Residual-Encoded-Distillation/"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
DEVICES=0

########### DATASET SETTINGS ############
DATASET=STL10
IMSIZE=128
BATCHSIZE=64
#########################################

DIR=$0
STUDENT=$1
TEACHER=$2
TEACHERPRETRAINED=$3

# kd
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_distill.py \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --model_s $STUDENT \
          --model_t $TEACHER --path_t $TEACHERPRETRAINED \
          --epochs 300 --learning_rate 1e-2 --lr_decay_epochs '180 240 270' --lr_decay_rate 0.1 \
          --distill kd -r 0.1 -a 0.9 -b 0 --trial 1
# FitNet
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_distill.py \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --model_s $STUDENT \
          --model_t $TEACHER --path_t $TEACHERPRETRAINED \
          --epochs 300 --learning_rate 1e-2 --lr_decay_epochs '180 240 270' --lr_decay_rate 0.1 \
          --distill hint -a 0 -b 100 --trial 1
# AT
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_distill.py \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --model_s $STUDENT \
          --model_t $TEACHER --path_t $TEACHERPRETRAINED \
          --epochs 300 --learning_rate 1e-2 --lr_decay_epochs '180 240 270' --lr_decay_rate 0.1 \
          --distill attention -a 0 -b 1000 --trial 1
# SP
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_distill.py \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --model_s $STUDENT \
          --model_t $TEACHER --path_t $TEACHERPRETRAINED \
          --epochs 300 --learning_rate 1e-2 --lr_decay_epochs '180 240 270' --lr_decay_rate 0.1 \
          --distill similarity -a 0 -b 3000 --trial 1
# CC
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_distill.py \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --model_s $STUDENT \
          --model_t $TEACHER --path_t $TEACHERPRETRAINED \
          --epochs 300 --learning_rate 1e-2 --lr_decay_epochs '180 240 270' --lr_decay_rate 0.1 \
          --distill correlation -a 0 -b 0.02 --trial 1
# VID
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_distill.py \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --model_s $STUDENT \
          --model_t $TEACHER --path_t $TEACHERPRETRAINED \
          --epochs 300 --learning_rate 1e-2 --lr_decay_epochs '180 240 270' --lr_decay_rate 0.1 \
          --distill vid -a 0 -b 1 --trial 1
# RKD
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_distill.py \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --model_s $STUDENT \
          --model_t $TEACHER --path_t $TEACHERPRETRAINED \
          --epochs 300 --learning_rate 1e-2 --lr_decay_epochs '180 240 270' --lr_decay_rate 0.1 \
          --distill rkd -a 0 -b 1 --trial 1
# PKT
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_distill.py \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --model_s $STUDENT \
          --model_t $TEACHER --path_t $TEACHERPRETRAINED \
          --epochs 300 --learning_rate 1e-2 --lr_decay_epochs '180 240 270' --lr_decay_rate 0.1 \
          --distill pkt -a 0 -b 30000 --trial 1
# AB
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_distill.py \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --model_s $STUDENT \
          --model_t $TEACHER --path_t $TEACHERPRETRAINED \
          --epochs 300 --learning_rate 1e-2 --lr_decay_epochs '180 240 270' --lr_decay_rate 0.1 \
          --distill abound -a 0 -b 1 --trial 1
# FT
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_distill.py \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --model_s $STUDENT \
          --model_t $TEACHER --path_t $TEACHERPRETRAINED \
          --epochs 300 --learning_rate 1e-2 --lr_decay_epochs '180 240 270' --lr_decay_rate 0.1 \
          --distill factor -a 0 -b 200 --trial 1
# FSP
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_distill.py \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --model_s $STUDENT \
          --model_t $TEACHER --path_t $TEACHERPRETRAINED \
          --epochs 300 --learning_rate 1e-2 --lr_decay_epochs '180 240 270' --lr_decay_rate 0.1 \
          --distill fsp -a 0 -b 50 --trial 1
# NST
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_distill.py \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --model_s $STUDENT \
          --model_t $TEACHER --path_t $TEACHERPRETRAINED \
          --epochs 300 --learning_rate 1e-2 --lr_decay_epochs '180 240 270' --lr_decay_rate 0.1 \
          --distill nst -a 0 -b 50 --trial 1
