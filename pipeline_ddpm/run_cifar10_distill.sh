DIR=$0
LOGDIR=$1
DISTCONFIG=$2
DEVICES=$3

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$DEVICES


PYTHONPATH="$(dirname $DIR)/":$PYTHONPATH \
python main_reed.py \
       --train --distill \
       --data_type cifar10 --img_size 32 --num_class 10 \
       --logdir $LOGDIR --total_steps 1000001 \
       --sample_step 100000 --save_step 200000 \
       --batch_size 128 \
       --cfg --conditional \
       --teacher_pretrain ./logs/cifar10/ddpm-teacher/ckpt_1000000.pt \
       --dist_config $DISTCONFIG --pool_strides "2,2,2,1"