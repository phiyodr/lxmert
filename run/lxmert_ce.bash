# The name of experiment
name=$1
echo `date`

# Create dirs and make backup
output=snap/cross_relationship_score/$name
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
    python src/pretrain/lxmert_pretrain.py \
    --taskMaskLM --taskObjPredict --taskMatched --taskPIaux  \
    --visualLosses obj,attr,feat \
    --train mscoco_minival_withpi \
    --tqdm --host_name $DGXNAME --evaluate_matchingscore --output $output ${@:2} 
echo `date`
