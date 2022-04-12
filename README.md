# README

In "Probing the Role of Positional Information in Vision-Language Models" we evaluate LXMERT models with different positional information input types using two probing and a downstream task. Later we add two new pre-training strategies (based on the probs) and report results for all experiments.

## Overview

This repository is a fork of [https://github.com/airsplay/lxmert/](https://github.com/airsplay/lxmert/).

* Source code is stored in `src`.
* Files to start training in `run`.
* Data should be stored in `data`.
* Training results (like models) are stored in `snap`.


## Preparation

* Please download the data as described [here](https://github.com/airsplay/lxmert/#gqa) and [here](https://github.com/airsplay/lxmert/#pre-training).
* Add the depth information and 9 MPE labels using `data/depth/README.md`.
* Consider using following Docker container:

```
docker run -it --gpus all --name plxmert -v plxmert:/root/plxmert/ nvcr.io/nvidia/pytorch:20.03-py3 bash
```


## Experiments 

Set parameters:

```
N_GPUS=8 		# 8 GPUs for pre-training
GPU_NUMBER=0	# 1 GPU for fine-tuning
EXPERIMENT_NAME = myLxmerdt
```


### Pre-training

Pre-training for plain and our version with different positional information input type. Models are stored in `snap/`.

```
bash run/plxmert_pretrain.bash $N_GPUS $EXPERIMENT_NAME ARGS
```

`ARGS`:

* `--report_cmm_acc` for plain version.
* `--task_pi_cl_cmm` and `--pi_aux_weight 10` for our version.
* `PI_INPUT_TYPE`: Add `--nopi` (no positional information), `--use_center` (x,y), `--use_bb` (x1,y1,x2,y2), or `--use_bb --use_d_med` (x1,y1,x2,y2,d).


### Mutual Position Evaluation (MPE)

Fine-tuning of PI head for MPE of 11k classification tasks for plain version. In our version this is done during pre-training.

```
bash run/plxmert_mpe.bash $GPU_NUMBER $EXPERIMENT_NAME --loadLXMERT snap/pretrain/$(EXPERIMENT_NAME)/BEST_EVAL_LOSS PI_INPUT_TYPE
```


### Contrastive Evaluation on PI using CMM (CE)

Evaluation of PI using cross-modality matching (CMM). 

```
bash run/plxmert_ce.bash $EXPERIMENT_NAME --valid DATA $PI_INPUT_TYPE --loadLXMERT snap/pretrain/$(EXPERIMENT_NAME)/BEST_EVAL_LOSS --matching_prob X 
```

* `X`: 0.5 for original LXMERT evaluation, `1` to permute all caption to a given image, or 0 not to permute caption.
* `DATA`: e.g. `mscoco_minival`




### Downstream Task Evaluation

Downstream task using GQA. Also report test subsets.

```
bash run/plxmert_gqa.bash $GPU_NUMBER $EXPERIMENT_NAME snap/pretrain/$(EXPERIMENT_NAME)/BEST_EVAL_LOSS PI_INPUT_TYPE 
```