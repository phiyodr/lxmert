# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sname', type=str)

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)
    # Predict topk results
    parser.add_argument('--topk', type=int, default=1)
    
    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)
    parser.add_argument('--dgxname', type=str)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')
    parser.add_argument('--saving_model_name', type=str, default=None,
                        help='Load model from loadLXMERT, then save under this name.')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')

    # LXMERT Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    parser.add_argument("--taskPIaux", dest='task_pi_aux', action='store_const', default=False, const=True)
    parser.add_argument("--pi_aux_weight", dest='pi_aux_weight', default=0.0, type=float)
    parser.add_argument("--downscale_other_losses_but_pi",  action='store_const', default=False, const=True)
    parser.add_argument("--pi_dropout_rate", dest='pi_dropout_rate', default=0.0, type=float)
    parser.add_argument("--gqa_dropout_rate", dest='gqa_dropout_rate', default=0.0, type=float)
    parser.add_argument("--report_pi_acc", action='store_const', default=False, const=True)
    parser.add_argument("--report_qa_acc", action='store_const', default=False, const=True)
    parser.add_argument("--pi_loss_only", action='store_const', default=False, const=True)
    parser.add_argument("--matching_prob", default=0.5, type=float)
    parser.add_argument("--task_pi_cl_cmm", action='store_const', default=False, const=True)
    parser.add_argument("--whole_coco_cmm_eval", action='store_const', default=False, const=True)
    parser.add_argument("--mscoco_only", action='store_const', default=False, const=True)
    parser.add_argument("--cmm_extra_weight",  default=1.0, type=float)

    parser.add_argument("--report_cmm_acc", action='store_const', default=False, const=True)
    parser.add_argument("--visual_weights", dest='visual_weights', default=1/0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', type=int, default=0) # used to be 0

    # Do not set this argument by you one. This arguement is required for torch.distributed.launch 
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary \
                        for using the torch.distributed.launch utility.")
    # Add wandb logging
    parser.add_argument("--wandb", action='store_const', default=False, const=True)
    parser.add_argument("--host_name", default=None, type=str)

    parser.add_argument("--evaluate", action='store_const', default=False, const=True)
    parser.add_argument("--pre_eval", action='store_const', default=False, const=True)
    parser.add_argument("--evaluate_matchingscore", action='store_const', default=False, const=True)
    parser.add_argument("--finetune_pi_head", action='store_const', default=False, const=True)
   
    # lxmerDt specific arguments
    parser.add_argument("--use_pkl", action='store_const', default=True, const=True)
    parser.add_argument("--dont_use_pkl", action='store_const', default=False, const=True)
    parser.add_argument("--old", action='store_const', default=False, const=True)
    parser.add_argument("--new", action='store_const', default=False, const=True)

    # old
    parser.add_argument("--center_only", action='store_const', default=False, const=True)
    parser.add_argument("--real_center_only", action='store_const', default=False, const=True)
    parser.add_argument("--area",  action='store_const', default=False, const=True)
    parser.add_argument("--depth_type", dest='depth_type', default=None, type=str)
    parser.add_argument("--quant",   action='store_const', default=False, const=True)
    parser.add_argument("--add_std",   action='store_const', default=False, const=True)
    parser.add_argument("--add_quantiles",   action='store_const', default=False, const=True)
    parser.add_argument("--log_depth",   action='store_const', default=False, const=True)
    parser.add_argument("--nopi",   action='store_const', default=False, const=True)
    
    #new
    parser.add_argument("--use_center",   action='store_const', default=False, const=True)
    parser.add_argument("--use_bb",   action='store_const', default=False, const=True)
    parser.add_argument("--use_area_rel",   action='store_const', default=False, const=True)
    parser.add_argument("--use_area_absolute",   action='store_const', default=False, const=True)
    parser.add_argument("--use_wh",   action='store_const', default=False, const=True)
    parser.add_argument("--use_d_med",   action='store_const', default=False, const=True)
    parser.add_argument("--use_d_mean",   action='store_const', default=False, const=True)
    parser.add_argument("--use_d_cntr",   action='store_const', default=False, const=True)
    parser.add_argument("--use_d_std",   action='store_const', default=False, const=True)
    parser.add_argument("--use_d_q25",   action='store_const', default=False, const=True)
    parser.add_argument("--use_d_q75",   action='store_const', default=False, const=True)
    parser.add_argument("--use_d_quant",   action='store_const', default=False, const=True)
    
    # only works for "old" settings:
    # permute and randmoize pos/depth info 
    parser.add_argument("--depth_zero",   action='store_const', default=False, const=True)
    parser.add_argument("--depth_one",   action='store_const', default=False, const=True)
    parser.add_argument("--depth_randiter",   action='store_const', default=False, const=True)
    parser.add_argument("--depth_iter",   action='store_const', default=False, const=True)
    parser.add_argument("--depth_permute",   action='store_const', default=False, const=True)
    parser.add_argument("--depth_randomize", action='store_const', default=False, const=True)

    parser.add_argument("--all_zero",   action='store_const', default=False, const=True)
    parser.add_argument("--all_randiter",   action='store_const', default=False, const=True)
    parser.add_argument("--all_iter",   action='store_const', default=False, const=True)
    parser.add_argument("--all_permute",     action='store_const', default=False, const=True)
    parser.add_argument("--all_randomize",   action='store_const', default=False, const=True)


    # Parse the arguments.
    args = parser.parse_args()
    if args.dont_use_pkl:
        args.use_pkl = False

    # pos_dim
    if args.center_only or args.real_center_only:
        pos_dim = 2
    else:
        pos_dim = 4
    if args.depth_type: pos_dim += 1
    if args.area: pos_dim += 1
    if args.add_std: pos_dim += 1
    if args.add_quantiles: pos_dim += 2

    if args.new:
        args.visual_pos_dim = 20 #pos_dim
    else:
        args.visual_pos_dim = pos_dim
    #print(f"Args: visual_pos_dim={args.visual_pos_dim}")
    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
print("-"*70)
print(args)
print(args.num_workers, type(args.num_workers))
print("-"*70)
