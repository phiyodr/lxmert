# The name of experiment
name=$2
echo `date`

# Create dirs and make backup
output=snap/pretrain_finetunepihead/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

echo $output

if [ "$HOST_HOSTNAME" ]
then
  DGXNAME=$HOST_HOSTNAME
else
  DGXNAME=$HOSTNAME
fi

echo Using GPU: $1
echo $name

# Pre-training
CUDA_VISIBLE_DEVICES=$1  PYTHONPATH=$PYTHONPATH:./src \
     python -m torch.distributed.launch --nproc_per_node=1 src/pretrain/lxmert_pretrain.py \
    --taskMaskLM --taskObjPredict --taskMatched --taskQA \
    --visualLosses obj,attr,feat \
    --wordMaskRate 0.15 --objMaskRate 0.15 \
    --train mscoco_train,mscoco_nominival,vgnococo --valid mscoco_minival \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --tqdm --host_name $DGXNAME \
    --epochs 1 --optim bert \
    --finetune_pi_head --report_pi_acc --taskPIaux --pi_aux_weight 1 \
    --pi_loss_only \
    --batchSize 256 --lr 1e-4  --gqa_dropout_rate 0.1 --pi_dropout_rate 0.1 \
    --pre_eval \
    --output $output ${@:3} 
echo `date`
