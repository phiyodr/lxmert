# The name of experiment
name=$2
echo `date`

# Create dirs and make backup
output=snap/pretrain/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash


if [ "$HOST_HOSTNAME" ]
then
  DGXNAME=$HOST_HOSTNAME
else
  DGXNAME=$HOSTNAME
fi

# Pre-training
PYTHONPATH=$PYTHONPATH:./src \
    python -m torch.distributed.launch --nproc_per_node=$1 src/pretrain/lxmert_pretrain.py \
    --taskMaskLM --taskObjPredict --taskMatched --taskQA \
    --visualLosses obj,attr,feat \
    --wordMaskRate 0.15 --objMaskRate 0.15 \
    --train mscoco_train,mscoco_nominival,vgnococo --valid mscoco_minival \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --fromScratch \
    --optim bert --epochs 20 \
    --batchSize 256 --lr 1e-4  --gqa_dropout_rate 0.1 \
    --taskPIaux --pi_dropout_rate 0.1 --report_pi_acc \
    --multiGPU  --tqdm --host_name $DGXNAME --sname $2 --output $output ${@:3}
echo `date`