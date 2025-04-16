DIR=$0
MODE=$1
LOGDIR=$2
DEVICES=$3

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$DEVICES


if [[ $MODE == single ]]
then
PYTHONPATH="$(dirname $DIR)/":$PYTHONPATH \
python main_reed.py \
       --train --parallel --nodistill \
       --data_type celebahq --data_path .cache/data/ --img_size 256 --num_class 0 \
       --logdir $LOGDIR --total_steps 300001 \
       --sample_step 100000 --save_step 50000 --sample_method uncond \
       --batch_size 32 \
       --nocfg --noconditional \
       --pool_strides "8,1,1,1"
elif [[ $MODE == distill ]]
then
DISTCONFIG=$4
PYTHONPATH="$(dirname $DIR)/":$PYTHONPATH \
python main_reed.py \
       --train --parallel --distill \
       --data_type celebahq --data_path .cache/data/ --img_size 256 --num_class 0 \
       --logdir $LOGDIR --total_steps 300001 \
       --sample_step 100000 --save_step 50000 --sample_method uncond \
       --batch_size 32 \
       --nocfg --noconditional \
       --dist_config $DISTCONFIG --pool_strides "8,1,1,1"
elif [[ $MODE == nodistill ]]
then
PYTHONPATH="$(dirname $DIR)/":$PYTHONPATH \
python main_reed.py \
       --train --parallel --nodistill \
       --data_type celebahq --data_path .cache/data_base/ --img_size 256 --num_class 0 \
       --logdir $LOGDIR --total_steps 300001 \
       --sample_step 100000 --save_step 50000 --sample_method uncond \
       --batch_size 32 \
       --nocfg --noconditional \
       --dist_config "./config/no_dist_config.yaml" --pool_strides "8,1,1,1"
fi
# CUDA_VISIBLE_DEVICES="0" python main_reed.py --train --data_type celebahq --data_path .cache/data/ --img_size 256 --num_class 0 --logdir ./logs/celebahq-256/ddpmx8/ --total_steps 300001 --batch_size 64 --nocfg --noconditional --pool_strides "8,1,1,1" --nodistill
