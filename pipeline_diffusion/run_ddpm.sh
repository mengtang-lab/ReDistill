##!/bin/bash
ROOT="/home/fangchen/WorkDir/Residual-Encoded-Distillation/"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

DATA_ROOT=$1
MODE=$2
CONFIG=$3
if [[ $MODE == vanilla ]]
then
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python main.py --flagfile $CONFIG --dataset_root $DATA_ROOT --train --noeval
elif [[ $MODE == reed ]]
then
PYTHONPATH="$(dirname $DIR)/":$ROOT:$PYTHONPATH \
python main_reed.py --flagfile $CONFIG --dataset_root $DATA_ROOT --train --noeval
fi