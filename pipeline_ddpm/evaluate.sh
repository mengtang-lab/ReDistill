DIR=$0
IMG_SIZE=$1
METRICS=$2
EXTRACTOR=$3
DATA_TYPE=$4
DATA_PATH=$5
GEN_NPY_CACHE=$6


export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$DEVICES


PYTHONPATH="$(dirname $DIR)/":$PYTHONPATH \
  python evaluate.py --img_size $IMG_SIZE \
  --metrics $METRICS --extractor $EXTRACTOR \
  --data_type $DATA_TYPE --data_path $DATA_PATH \
  --imb_factor 1.0 --gen_npy_cache $GEN_NPY_CACHE