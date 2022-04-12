# The name of this experiment.
name=$2
echo `date`

# Save logs and models under snap/gqa; make backup.
output=snap/gqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

if [ "$HOST_HOSTNAME" ]
then
  DGXNAME=$HOST_HOSTNAME
else
  DGXNAME=$HOSTNAME
fi


# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python -u src/tasks/gqa2.py \
    --train train --valid valid --test testdev \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --sname $2 \
    --loadLXMERTQA $3 \
    --batchSize 32 --optim bert --lr 1e-5 --epochs 5 \
    --dgxname $DGXNAME \
    --tqdm --output $output ${@:4}
echo `date`
