DIR=$0
LOGDIR=$1
FID_CACHE=$2
FEATS_CACHE=$3
CKPT_STEP=$4
DEVICES=$5
NUM_IMAGES=$6
BATCHSIZE=$7
DDIM_SKIP_STEP=$8

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$DEVICES


for i in `seq 0 5 20`
do
  omega=$(echo "scale=2; 0.1 * $i" | bc)
  echo "omega: $omega"
    PYTHONPATH="$(dirname $DIR)/":$PYTHONPATH \
    python  main.py --flagfile $LOGDIR/flagfile.txt \
            --notrain --eval --parallel \
            --conditional --cfg \
            --logdir $LOGDIR \
            --fid_cache $FID_CACHE \
            --feats_cache $FEATS_CACHE \
            --ckpt_step $CKPT_STEP \
            --num_images $NUM_IMAGES --batch_size $BATCHSIZE \
            --sample_method "ddim" \
            --ddim_skip_step $DDIM_SKIP_STEP \
            --omega $omega \
            --noprd --noimproved_prd
done