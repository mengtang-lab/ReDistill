##!/bin/bash
ROOT="/home/fangchen/WorkDir/Residual-Encoded-Distillation/"


export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1"
DEVICES=0

########### DATASET SETTINGS ############
DATASET=STL10
IMSIZE=128
BATCHSIZE=8
#########################################

DIR=$0
MODE=$1
STUDENT=$2

if [[ $MODE == single ]]
then
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_single.py \
          --root ./ --seed 0 --devices $DEVICES \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --architecture $STUDENT --pretrained '' \
          --epochs 300 --learning_rate 1e-2 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD \
          --log
elif [[ $MODE == quantize ]]
then
PRETRAINED=$3
TEACHER=$4
TEACHERPRETRAINED=$5
CONFIG=$6
DISTPRETRAINED=$7
QUANT=$8
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_quant.py \
          --root ./ --seed 0 --devices $DEVICES \
          --dataset $DATASET --im_size $IMSIZE --batch_size 8 \
          --architecture $STUDENT --pretrained $PRETRAINED \
          --teacher $TEACHER --teacher_pretrained $TEACHERPRETRAINED \
          --dist_config $CONFIG --dist_pretrained $DISTPRETRAINED \
          --epochs 1 --learning_rate 1e-2 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD \
          --log --quantize $QUANT
#bash run_classification.sh quantize resnext18-4-11221 ./ckpt/resnext18-4-11221_stl10_imsize128_batchsize8_lr0.01_optimizerSGD.pth resnext18-1-21222 ./ckpt/resnext18-1-21222_stl10_imsize128_batchsize8_lr0.001_optimizerAdam.pth ./configs/resnext18-red-rks5.yaml ./ckpt/resnext18-4-11221_resnext18-red-rks5.yaml_stl10_imsize128_batchsize8_lr0.01_optimizerSGD.pth ptq
elif [[ $MODE == distill ]]
then
TEACHER=$3
TEACHERPRETRAINED=$4
CONFIG=$5
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python train_distill.py \
          --root ./ --seed 0 --devices $DEVICES \
          --dataset $DATASET --im_size $IMSIZE --batch_size $BATCHSIZE \
          --architecture $STUDENT \
          --teacher $TEACHER --teacher_pretrained $TEACHERPRETRAINED \
          --dist_config $CONFIG --dist_pretrained '' \
          --epochs 300 --learning_rate 1e-2 --lr_interval '0.6 0.8 0.9' --lr_reduce 5 --optimizer SGD \
          --log
fi
