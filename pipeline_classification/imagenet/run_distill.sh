#!/bin/bash
ROOT="/home/fangchen/WorkDir/Residual-Encoded-Distillation/"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

DIR=$0
MODE=$1
DATA_ROOT=$2
CONFIG=$3

if [[ $MODE == dist ]]
then
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python tools/train.py --data_path $DATA_ROOT --cfg $CONFIG --devices 0 \
#                      --resume
elif [[ $MODE == reed ]]
then
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python tools/train_reed.py --data_path $DATA_ROOT --cfg $CONFIG --devices 0 \
#                           --resume
fi