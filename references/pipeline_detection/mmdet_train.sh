#!/usr/bin/env bash
ROOT=/home/fangchen/WorkDir/Residual-Encoded-Distillation/
CONFIG=$1

PYTHONPATH="$(dirname $0)/":"$ROOT":$PYTHONPATH \
python ./mmdetection/tools/train.py $CONFIG