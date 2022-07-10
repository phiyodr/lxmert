# Probing the Role of Positional Information in Vision-Language Models

All details: [https://www.unibw.de/vis-en/naacl2022](https://www.unibw.de/vis-en/naacl2022)

In "Probing the Role of Positional Information in Vision-Language Models" we evaluate LXMERT models with different positional information (PI) input types using two probing and a downstream task. Later we add two new pre-training strategies (based on the probs) and report results for all experiments.


## Overview

This repository is a fork of [https://github.com/airsplay/lxmert/](https://github.com/airsplay/lxmert/). The analysis is based on [LXMERT](https://aclanthology.org/D19-1514.pdf).

* Source code is stored in `src`.
* Files to start training in `run`.
* Data should be stored in `data`.
* Training results (like models) are stored in `snap`.


## Preparation

* Please download the data as described [here](https://github.com/airsplay/lxmert/#gqa) and [here](https://github.com/airsplay/lxmert/#pre-training).
* Add the Depth Information and 9 Mutual Positions labels using `data/depth/README.md`.
* Consider using following Docker container:

```
docker run -it --gpus all --ipc=host --name plxmert -v plxmert:/root/plxmert/ nvcr.io/nvidia/pytorch:20.03-py3 bash
```




## Experiments 

Set parameters:

```
N_GPUS=8 	# 8 GPUs for pre-training (takes about 41 hours)
GPU_NUMBER=0 	# 1 GPU for fine-tuning (takes about 11 hours)
EXPERIMENT_NAME = mylxmert 
```


### Pre-training

Pre-training for original and our version with different positional information input types. Models are stored in `snap/`.

```
bash run/plxmert_pretrain.bash $N_GPUS $EXPERIMENT_NAME ARGS
```

`ARGS`:

* `--report_cmm_acc` for original version.
* `--task_pi_cl_cmm` and `--pi_aux_weight 10` for our version.
* `PI_INPUT_TYPE`: Add `--nopi` (no positional information), `--use_center` (x,y), `--use_bb` (x1,y1,x2,y2), or `--use_bb --use_d_med` (x1,y1,x2,y2,d).


### Mutual Position Evaluation (MPE)

Fine-tuning of PI head for MPE of 11k classification tasks for original version. In our version this is done during pre-training.

```
bash run/lxmert_mpe.bash $GPU_NUMBER $EXPERIMENT_NAME --loadLXMERT snap/pretrain/$(EXPERIMENT_NAME)/BEST_EVAL_LOSS PI_INPUT_TYPE
```


### Contrastive Evaluation on PI using CMM (CE)

Evaluation of PI using cross-modality matching (CMM). 

```
bash run/lxmert_ce.bash $EXPERIMENT_NAME --valid DATA $PI_INPUT_TYPE --loadLXMERT snap/pretrain/$(EXPERIMENT_NAME)/BEST_EVAL_LOSS --matching_prob X 
```

* `X`: 0.5 for original LXMERT evaluation, `1` to permute all captions to a given image, or 0 not to permute captions.
* `DATA`: e.g. `mscoco_minival`




### Downstream Task Evaluation

Downstream task using GQA. This also reports test subsets.

```
bash run/lxmert_gqa.bash $GPU_NUMBER $EXPERIMENT_NAME snap/pretrain/$(EXPERIMENT_NAME)/BEST_EVAL_LOSS PI_INPUT_TYPE 
```


## Checkpoints

You can download pre-trained models:

|                 | No positional information                                                                                   | `--use_center`                                                                                            | `--use_bb`                                                                                                      | `--use_bb --use_d_med`                                                                                           |
|-----------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Plain LXMERT    | [`lxmert_nopi`](https://drive.google.com/file/d/1AD3zpUYQL3gT8ycXRIK27F0gy1W4i_dJ/view?usp=sharing)                                                    | [`lxmert_xy`](https://drive.google.com/file/d/1bAE9j0dmuhRCUMRI6LDRveEcKgbamFxU/view?usp=sharing)                                                            | [`lxmert_x1y1x2y2`](https://drive.google.com/file/d/1KFJni4TREYbp0J0B52bAYKYwtfsm2oFS/view?usp=sharing)                                        | [`lxmert_x1y1x2y2d_`](https://drive.google.com/file/d/1nwJ6yUS157a5we4U9VPO55cSyN1gNLGi/view?usp=sharing)                                       |
| With PIP and CL | [`lxmert_nopi_pipcl`](https://drive.google.com/drive/folders/10DbTnZpGSuHsYWgA0j0HIwQAC1ZOaNfH?usp=sharing) | [`lxmert_xy_pipcl`](https://drive.google.com/drive/folders/1a3HUDYYLFMfKtBUey84Q-PjK-wFrtztp?usp=sharing) | [`lxmert_x1y1x2y2_pipcl`](https://drive.google.com/drive/folders/1LlD0rGrzZYkdKAzwQgnwO6IHLJRpInCN?usp=sharing) | [`lxmert_x1y1x2y2d_pipcl`](https://drive.google.com/drive/folders/1kr1J1HD1hxjtQhuc1xrUIEjBVLlxFQit?usp=sharing) |


Place the models at `snap/pretrain/$(EXPERIMENT_NAME)/BEST_EVAL_LOSS_LXRT.pth`

