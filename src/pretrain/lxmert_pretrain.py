# coding=utf-8
# Copyleft 2019 project LXRT.

import collections
import os
import time
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import numpy as np

from param import args
from pretrain.lxmert_data import InputExample, LXMERTDataset, LXMERTTorchDataset, LXMERTEvaluator
from lxrt.entry import set_visual_config
from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTPretraining
from torchmetrics import Accuracy, ConfusionMatrix

import wandb

DataTuple = collections.namedtuple("DataTuple", 'dataset torchdset loader evaluator')

import pandas as pd
from sklearn.metrics import confusion_matrix 

def df_logit2prob(df, negativ=False):
    df.columns = ["is_match", "logit0", "logit1"]
    print(df.shape)
    if negativ:
        df["is_match"] = 0.0
    X = df[["logit0", "logit1"]].values
    X = torch.from_numpy(X)
    m = nn.Softmax(dim=1)
    output = m(X)
    df_pred = pd.DataFrame(output.numpy())
    df = pd.concat([df, df_pred], axis=1)
    df.columns = ["is_match", "logit0", "logit1", "prob0", "prob1"]
    df["y_pred"] = (df["prob1"] > 0.5).astype(float)
    df["correct"] = (df["is_match"] == df["y_pred"]).astype(float)
    return df


def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False, topk=-1, mscoco_only=False) -> DataTuple:
    # Decide which QA datasets would be used in pre-training.
    # Options: vqa, gqa, visual7w
    # Note: visual7w is a part of vgqa, we take the name here.
    #print(f"===> get_tuple for {splits} in rank {args.local_rank}.")
    qa_sets = args.qa_sets
    if qa_sets is not None:
        qa_sets = set(qa_set.lower().strip() for qa_set in qa_sets.split(","))

    # Build dataset, data loader, and evaluator.
    dset = LXMERTDataset(splits, qa_sets=qa_sets)
    tset = LXMERTTorchDataset(dset, args.use_pkl, args.center_only, args.depth_type, args.area, topk, mscoco_only, args.matching_prob, args.task_pi_cl_cmm)

    # DistributedSampler for distributed training, pass to DataLoader
    data_sampler = torch.utils.data.distributed.DistributedSampler(tset) if args.multiGPU else None
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        collate_fn=lambda x: x,
        drop_last=drop_last, pin_memory=True, sampler=data_sampler
    )

    evaluator = LXMERTEvaluator(dset)
    print()

    return DataTuple(dataset=dset, torchdset=tset, loader=data_loader, evaluator=evaluator)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, input_mask, segment_ids, lm_label_ids,
                 visual_feats, obj_labels,
                 is_matched, ans, pi_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids

        self.visual_feats = visual_feats #  (feat, boxes/objs_pos) 
        self.obj_labels = obj_labels

        self.is_matched = is_matched

        self.ans = ans

        self.pi_labels = pi_labels


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with probability
        ratio = args.word_mask_rate
        if prob < ratio:
            prob /= ratio

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def random_feat(train_tuple, feats):
    mask_feats = feats.copy()
    feat_mask = np.zeros(len(feats), dtype=np.float32)
    for i in range(len(feats)):
        prob = random.random()
        # mask token with probability
        if prob < args.obj_mask_rate:
            prob /= args.obj_mask_rate

            # 80% randomly change token to zero feat
            if prob < 0.8:
                mask_feats[i, :] = 0.

            # 10% randomly change token to random feat
            elif prob < 0.9:
                mask_feats[i, :] = train_tuple.torchdset.random_feat()
            # -> rest 10% randomly keep current feat

            # Need to predict this feat
            feat_mask[i] = 1.

    return mask_feats, feat_mask


def convert_example_to_features(train_tuple, example: InputExample, max_seq_length, tokenizer)->InputFeatures:
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens = tokenizer.tokenize(example.sent.strip())

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Ge random words
    masked_tokens, masked_label = random_word(tokens, tokenizer)

    # concatenate lm labels and account for CLS, SEP, SEP
    masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    # Mask & Segment Word
    lm_label_ids = ([-1] + masked_label + [-1])
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    feat, boxes = example.visual_feats
    obj_labels, obj_confs = example.obj_labels
    attr_labels, attr_confs = example.attr_labels

    # Mask Image Features:
    masked_feat, feat_mask = random_feat(train_tuple, feat)
    #masked_pos, pos_mask = random_feat(train_tuple, boxes)
    #pos, pos_mask = boxes, np.ones(len(boxes), dtype=np.float32)
    pi_labels = example.pi_labels


    # QA answer label
    if example.label is None or len(example.label) == 0 or example.is_matched != 1:
        # 1. No label 2. Label is pruned 3. unmatched visual + language pair
        ans = -1
    else:
        keys, values = zip(*example.label.items())
        if len(keys) == 1:
            ans = keys[0]
        else:
            value_sum = sum(values)
            prob = [value / value_sum for value in values]
            choice = np.random.multinomial(1, prob).argmax()
            ans = keys[choice]

    features = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        lm_label_ids=lm_label_ids,
        visual_feats=(masked_feat, boxes),
        obj_labels={
            'obj': (obj_labels, obj_confs),
            'attr': (attr_labels, attr_confs),
            'feat': (feat, feat_mask),
            #'pos': (pos, pos_mask)
        },
        is_matched=example.is_matched,
        ans=ans, pi_labels=pi_labels
    )
    return features


LOSSES_NAME = ('Mask_LM', 'Matched', 'Obj', 'Attr', 'Feat', 'QA', "PI_Aux")


class LXMERT:
    def __init__(self, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.multiGPU = args.multiGPU
        if self.multiGPU:
            # Init for DistributedDataParallel
            dist.init_process_group(backend='nccl') # Init process with default init_method='env://'
            #torch.cuda.set_device(args.local_rank) # Set default device for this process
            #torch.manual_seed(0)
            #self.model.cuda() # Move model to currect cuda device  
            #self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],  output_device=args.local_rank) # Wrap the model. 
            print(f"I am local_rank={args.local_rank}, get_rank={torch.distributed.get_rank()}, get_world_size ={torch.distributed.get_world_size()}")
        else:
            print("Single GPU-Training!")
        #Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        if self.multiGPU:
            torch.distributed.barrier()
        else:
            args.local_rank == 0
        # Build model
        set_visual_config(args)
        self.model = LXRTPretraining.from_pretrained(
            "bert-base-uncased",
            task_mask_lm=args.task_mask_lm,
            task_obj_predict=args.task_obj_predict,
            task_matched=args.task_matched,
            task_qa=args.task_qa,
            visual_losses=args.visual_losses,
            num_answers=9500,#train_tuple.dataset.answer_table.num_answers
            visual_pos_dim=args.visual_pos_dim,
            task_pi_aux=args.task_pi_aux,
            pi_aux_weight = args.pi_aux_weight,
            gqa_dropout_rate=args.gqa_dropout_rate,
            pi_dropout_rate=args.pi_dropout_rate,
            pi_loss_only=args.pi_loss_only,
            downscale_other_losses_but_pi=args.downscale_other_losses_but_pi,
            cmm_extra_weight = args.cmm_extra_weight
        )
        if self.multiGPU:
            torch.distributed.barrier()
        self.task_qa = args.task_qa
        self.report_pi_acc = args.report_pi_acc
        self.report_qa_acc = args.report_qa_acc
        self.report_cmm_acc = args.report_cmm_acc
            
        # Weight initialization and loading
        if args.from_scratch:
            print("Train from Scratch: re-initialize all BERT weights.")
            self.model.apply(self.model.init_bert_weights)
        if args.load is not None:
            self.load(args.load)
        if args.load_lxmert is not None:
            # Load lxmert would not load the answer head.
            self.load_lxmert(args.load_lxmert)
            print("### load_lxmert from init")
            for name, param in self.model.named_parameters():
                print(name, param.requires_grad, param.shape)
            print("### load_lxmert from init DONE")
        
        # Init for DistributedDataParallel
        #dist.init_process_group(backend='nccl') # Init process with default init_method='env://'
        if self.multiGPU:
            torch.cuda.set_device(args.local_rank) # Set default device for this process
            torch.manual_seed(0)
            if args.local_rank == 0:
                print(self.model)
            self.model.cuda() # Move model to currect cuda device  
        else:
            torch.manual_seed(0)
            self.model.cuda() # Move model to currect cuda device  
        if self.multiGPU:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],  output_device=args.local_rank, find_unused_parameters=True) # Wrap the model
        #print(f"I am local_rank={args.local_rank}, get_rank={torch.distributed.get_rank()}, get_world_size ={torch.distributed.get_world_size()}")
        if args.wandb and args.local_rank == 0:
            wandb.init(project=f"lxmerdt2")
            wconfig = wandb.config
            wconfig.args = args
            if args.center_only:
                lxmerdt_name = "xy"
            else:
                lxmerdt_name = "xywh"
            if args.area:
                lxmerdt_name += "a"
            if args.depth_type:
                tpe = args.depth_type.split("_")[1]
                tpe = "_" + tpe
                lxmerdt_name += tpe

            wconfig.sname = args.sname
            wconfig.lxmerdt_name = lxmerdt_name
            wconfig.tiny = args.tiny
            wconfig.train_data = args.train
            wconfig.valid_data = args.valid
            wconfig.n_gpus = torch.distributed.get_world_size()
            wconfig.host_name = args.host_name
            wconfig.bs = args.batch_size
            wconfig.ebs = args.batch_size * torch.distributed.get_world_size()
            wconfig.lr = args.lr
            #config.d_center_only = args.center_only
            #config.d_area = args.area
            #config.d_depth_type = args.depth_type
            wconfig.task_pi_aux = args.task_pi_aux
            wconfig.pi_aux_weight = args.pi_aux_weight
            wconfig.report_pi_acc = args.report_pi_acc
            wconfig.report_qa_acc = args.report_qa_acc
            wconfig.report_cmm_acc = args.report_cmm_acc
            wconfig.gqa_dropout_rate = args.gqa_dropout_rate
            wconfig.pi_dropout_rate = args.pi_dropout_rate
            wconfig.task_pi_cl_cmm = args.task_pi_cl_cmm
            wconfig.cmm_extra_weight = args.cmm_extra_weight
            wconfig.visual_weights = args.visual_weights
            wconfig.old = args.old
            wconfig.new = args.new

            # old
            wconfig.center_only = args.center_only
            wconfig.real_center_only = args.real_center_only
            wconfig.area = args.area
            wconfig.depth_type = args.depth_type
            wconfig.quant = args.quant
            wconfig.add_std = args.add_std
            wconfig.add_quantiles = args.add_quantiles
            wconfig.log_depth = args.log_depth
            wconfig.nopi = args.nopi

            #new
            wconfig.use_center = args.use_center
            wconfig.use_bb = args.use_bb
            wconfig.use_area_rel = args.use_area_rel
            wconfig.use_area_absolute = args.use_area_absolute
            wconfig.use_wh = args.use_wh
            wconfig.use_d_med = args.use_d_med
            wconfig.use_d_mean = args.use_d_mean
            wconfig.use_d_cntr = args.use_d_cntr
            wconfig.use_d_std = args.use_d_std
            wconfig.use_d_q25 = args.use_d_q25
            wconfig.use_d_q75 = args.use_d_q75
            wconfig.use_d_quant = args.use_d_quant


        if self.multiGPU:
            torch.distributed.barrier()
        if args.report_pi_acc:
            self.train_pi_accs = Accuracy(average="none", num_classes=36*36*9).cuda()
            self.train_pi_acc  = Accuracy(num_classes=36*36*9).cuda()
            self.valid_pi_accs = Accuracy(average="none", num_classes=36*36*9).cuda()
            self.valid_pi_acc  = Accuracy(num_classes=36*36*9).cuda()
        if self.report_qa_acc:
            self.train_qa_accs = Accuracy(average="none", num_classes=9500).cuda()
            self.train_qa_acc  = Accuracy(num_classes=9500).cuda()
            self.valid_qa_accs = Accuracy(average="none", num_classes=9500).cuda()
            self.valid_qa_acc  = Accuracy(num_classes=9500).cuda()
        if self.report_cmm_acc:
            self.train_cmm_acc = Accuracy(num_classes=2).cuda()
            self.valid_cmm_acc = Accuracy(num_classes=2).cuda()
            self.train_cmm_cfm = ConfusionMatrix(num_classes=2).cuda()
            self.valid_cmm_cfm = ConfusionMatrix(num_classes=2).cuda()

            
    def train_pihead_only(self):
        print("===Freeze===")
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if name.split(".")[0] == "pi_head":
                param.requires_grad = True
            print(name, param.requires_grad)
            
    def train_heads_only(self):
        print("===Freeze===")
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            if name.split(".")[0] == "bert":
                param.requires_grad = True
            print(name, param.requires_grad)
            
        
    def forward(self, train_tuple, examples):
        train_features = [convert_example_to_features(train_tuple, example, self.max_seq_length, self.tokenizer)
                          for example in examples]

        # language Inputs
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()
        
        # Visual Inputs
        feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features])).cuda()
        pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features])).cuda()

        # Language Prediction
        lm_labels = torch.tensor([f.lm_label_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Prediction
        obj_labels = {}
        #for key in ('obj', 'attr', 'feat', 'pos'):
        for key in ('obj', 'attr', 'feat'):
            visn_labels = torch.from_numpy(np.stack([f.obj_labels[key][0] for f in train_features])).cuda()
            visn_mask = torch.from_numpy(np.stack([f.obj_labels[key][1] for f in train_features])).cuda()
            assert visn_labels.size(0) == visn_mask.size(0) and visn_labels.size(1) == visn_mask.size(1)
            obj_labels[key] = (visn_labels, visn_mask)

        # PI labels
        pi_labels = torch.from_numpy(np.stack([f.pi_labels for f in train_features])).cuda()


        # Joint Prediction
        matched_labels = torch.tensor([f.is_matched for f in train_features], dtype=torch.long).cuda()
        ans = torch.from_numpy(np.stack([f.ans for f in train_features])).cuda()

        """
        forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None, matched_label=None, ans=None):
        """
        loss, losses, ans_logit, res_dict = self.model(
            input_ids, segment_ids, input_mask, lm_labels,
            feats, pos, obj_labels, matched_labels, ans, pi_labels
        )

        res_dict["matched_labels"] = matched_labels
        return loss, losses.detach().cpu(), ans_logit, res_dict

    def train_batch(self, optim, train_tuple, batch, use_marked=True):
        optim.zero_grad()
        loss, losses, ans_logit, res_dict = self.forward(train_tuple, batch)
        if use_marked:
            marked = res_dict["ans"] > 0

        if args.report_pi_acc:
            res_dict["batch_pi_acc"] = self.train_pi_acc(res_dict["pi_scores"], res_dict["pi_labels"].int()).detach().cpu().numpy().item()
            self.train_pi_accs(res_dict["pi_scores"], res_dict["pi_labels"].int())
        if use_marked and args.report_qa_acc:
            if marked.sum().detach().cpu().numpy() > 0:
                res_dict["batch_qa_acc"] = self.train_qa_acc(res_dict["answer_score"][marked].argmax(dim=1), res_dict["ans"][marked]).detach().cpu().numpy().item()
                self.train_qa_accs(res_dict["answer_score"][marked].argmax(dim=1), res_dict["ans"][marked])
            else:
                res_dict["batch_qa_acc"] = 0
        if args.report_cmm_acc:
            res_dict["batch_cmm_acc"] = self.train_cmm_acc(res_dict["cmm_scores"], res_dict["cmm_labels"].int()).detach().cpu().numpy().item()
            res_dict["batch_cmm_cfm"] = self.train_cmm_cfm(res_dict["cmm_scores"], res_dict["cmm_labels"].int()).detach().cpu().numpy()
            
        if args.multiGPU:
            loss = loss.mean()
            losses = losses.mean(0)
        else:
            loss = loss
            losses = losses        
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        optim.step()

        return loss.item(), losses.cpu().numpy(), ans_logit, res_dict

    def valid_batch(self, train_tuple, batch, use_marked=False):
        with torch.no_grad():
            loss, losses, ans_logit, res_dict = self.forward(train_tuple, batch)
            if use_marked:
                marked = res_dict["ans"] > 0
            if args.report_pi_acc: 
                res_dict["batch_pi_acc"] = self.valid_pi_acc(res_dict["pi_scores"], res_dict["pi_labels"].int()).detach().cpu().numpy().item()
                self.valid_pi_accs(res_dict["pi_scores"], res_dict["pi_labels"].int())
            if use_marked and self.report_qa_acc:
                if marked.sum().detach().cpu().numpy() > 0:
                    res_dict["batch_qa_acc"] = self.valid_qa_acc(res_dict["answer_score"][marked].argmax(dim=1), res_dict["ans"][marked]).detach().cpu().numpy().item()
                    self.valid_qa_accs(res_dict["answer_score"][marked].argmax(dim=1), res_dict["ans"][marked])
            else:
                res_dict["batch_qa_acc"] = 0
            if args.report_cmm_acc:
                res_dict["batch_cmm_acc"] = self.valid_cmm_acc(res_dict["cmm_scores"], res_dict["cmm_labels"].int()).detach().cpu().numpy().item()
                res_dict["batch_cmm_cfm"] = self.valid_cmm_cfm(res_dict["cmm_scores"], res_dict["cmm_labels"].int()).detach().cpu().numpy()
          
            if args.multiGPU:
                loss = loss.mean()
                losses = losses.mean(0)            
            else:
                loss = loss.mean()
                losses = losses.mean(0)
        return loss.item(), losses.cpu().numpy(), ans_logit, res_dict

    def train(self):

        # Data loader
        train_tuple = get_tuple(args.train, args.batch_size, shuffle=False, drop_last=True, mscoco_only=False)
        print(f"Train data loaded for process {args.local_rank}.")
        valid_batch_size = 2048 if args.multiGPU else 512
        valid_batch_size = 256 if args.multiGPU else 256
        eval_tuple = get_tuple(args.valid, valid_batch_size, shuffle=False, drop_last=False, topk=5000, mscoco_only=False)
        train_loader = train_tuple.loader
        print("train_tuple.dataset.answer_table.num_answers", train_tuple.dataset.answer_table.num_answers)
        
        # Optimizer
        from lxrt.optimization import BertAdam
        batch_per_epoch = len(train_loader)
        t_total = int(batch_per_epoch * args.epochs)
        warmup_ratio = 0.05
        warmup_iters = int(t_total * warmup_ratio)
        print("Batch per epoch: %d" % batch_per_epoch)
        print("Total Iters: %d" % t_total)
        print("Warm up Iters: %d" % warmup_iters)
        optim = BertAdam(self.model.parameters(), lr=args.lr, warmup=warmup_ratio, t_total=t_total)

        # Eval
        if args.multiGPU:
            torch.distributed.barrier()
        if args.pre_eval:
            avg_eval_loss = self.evaluate_epoch(eval_tuple, train_tuple, iters=-1)
        if args.multiGPU:
            torch.distributed.barrier()

        # Train
        best_eval_loss = 9595.
        for epoch in range(args.epochs):
            print("#################")
            start_time = time.time()
            print("# Epoch", epoch, start_time)
            print("################")
            # Train
            self.model.train()
            total_loss = 0.
            total_losses = 0.
            uid2ans = {}
            train_step = 0
            for batch in tqdm(train_loader, total=len(train_loader), desc=f"Training epoch: {epoch}"):
                train_step += 1

                step_start_time = time.time()
                loss, losses, logit, res_dict = self.train_batch(optim, train_tuple, batch)
                total_loss += loss
                total_losses += losses

                if args.task_qa:
                    score, label = logit.max(1)
                    for datum, l in zip(batch, label.cpu().numpy()):
                        uid = datum.uid
                        ans = train_tuple.dataset.answer_table.id2ans(l)
                        uid2ans[uid] = ans

                if args.wandb and args.local_rank == 0:
                    step_duration = time.time()-step_start_time
                    local_obs_per_sec = args.batch_size / step_duration
                    if args.multiGPU:
                        total_obs_per_sec = args.batch_size *  torch.distributed.get_world_size() / step_duration
                    else:
                        total_obs_per_sec = -9
                    wandb.log({"Stat/epochs": epoch, "Stat/rel_steps": train_step/len(train_loader), "Stat/local_obs_per_sec": local_obs_per_sec, "Stat/total_obs_per_sec": total_obs_per_sec})
                    if args.report_pi_acc:
                        wandb.log({"training/PI_Acc": res_dict["batch_pi_acc"]})
                    if args.report_qa_acc:
                        wandb.log({"training/QA_Acc": res_dict["batch_qa_acc"]})
                    if args.report_cmm_acc:
                        wandb.log({"training/CMM_Acc": res_dict["batch_cmm_acc"]})
                    wandb.log({"training/loss": loss, "training/total_loss": total_loss/train_step})
                if args.multiGPU:
                    torch.distributed.barrier()
            
            # end of batch
            if args.wandb and args.local_rank == 0:
                wandb.log({"loss/train": loss, "total_loss/train": total_loss/train_step})
            if self.multiGPU:
                torch.distributed.barrier()
            

            if args.report_pi_acc:
                pi_acc  = self.train_pi_acc.compute()
                pi_accs = self.train_pi_accs.compute()
            if self.task_qa and self.report_qa_acc:
                qa_acc  = self.train_qa_acc.compute()
                qa_accs = self.train_qa_accs.compute()
            if args.report_cmm_acc:
                cmm_acc  = self.train_cmm_acc.compute()
                cmm_cfm  = self.train_cmm_cfm.compute()
                print("CMM_Acc_train", cmm_acc.detach().cpu().numpy().item())
                print("CMM_Cfm_train", cmm_cfm.detach().cpu().numpy())
                if args.wandb and (args.local_rank==0):
                    wandb.log({"CMM_Acc_train": cmm_acc.detach().cpu().numpy().item()}) 
                    wandb.log({"CMM_Cfm_train": cmm_cfm.detach().cpu().numpy()}) 
                
            if (args.local_rank == 0) and args.report_pi_acc:
                print("-"*50)
                print("Train epoch:", epoch, "pi_accs and min", pi_accs, pi_accs.min(), pi_accs.mean()) # torch.quantile(pi_accs, torch.tensor(0.05), dim=0, keepdim=False))
                all_accs = np.nan_to_num(pi_accs.detach().cpu().numpy(), nan=1.0)
                print("Train epoch:", epoch, "pi_accs and min", all_accs, all_accs.min(), all_accs.mean()) # torch.quantile(pi_accs, torch.tensor(0.05), dim=0, keepdim=False))
                row = np.expand_dims(np.concatenate(
                        (  [epoch], all_accs),
                        axis=0), 0)
                csv_path = os.path.join(args.output, "pi_accs_train.csv")
                print("Save in", csv_path)
                with open(csv_path, 'ab') as fff:
                    np.savetxt(fff, row, fmt="%.4f")
                print("-"*50)  
                
                if args.wandb:
                    wandb.log({"PI_Acc_train/mean": pi_acc.detach().cpu().numpy().item(), 
                        "PI_Acc_train/std": np.std(np.nan_to_num(pi_accs.detach().cpu().numpy(), nan=1.0)), 
                        "PI_Acc_train/min": np.min(np.nan_to_num(pi_accs.detach().cpu().numpy(), nan=1.0)),
                        "PI_Acc_train/q05": np.quantile(np.nan_to_num(pi_accs.detach().cpu().numpy(), nan=1.0), q=0.05),
                        "PI_Acc_train/q25": np.quantile(np.nan_to_num(pi_accs.detach().cpu().numpy(), nan=1.0), q=0.25)})


            print(args.local_rank, "The training loss for Epoch %d is %0.4f" % (epoch, total_loss / batch_per_epoch), "Epoch training time %0.2f minutes." % ((time.time()-start_time)/60) )
            losses_str = "The losses are "
            for name, loss in zip(LOSSES_NAME, total_losses):
                losses_str += "%s: %0.4f " % (name, loss / batch_per_epoch)

                if args.wandb and args.local_rank == 0:
                    train_losses_names = [i + "/train" for i in LOSSES_NAME]
                    wandb.log(dict(zip(train_losses_names, total_losses/len(train_loader))))
                if self.multiGPU:
                    torch.distributed.barrier()

            if self.multiGPU:
                torch.distributed.barrier()
            if args.task_qa:
                train_tuple.evaluator.evaluate(uid2ans, pprint=True)

            if self.report_pi_acc:
                self.train_pi_acc.reset()
                self.train_pi_accs.reset()
            if self.report_qa_acc:
                self.train_qa_acc.reset()
                self.train_qa_accs.reset()
            if self.report_cmm_acc:
                self.train_cmm_acc.reset()
                self.train_cmm_cfm.reset()
                
            # Eval
            if args.multiGPU:
                torch.distributed.barrier()
            avg_eval_loss = self.evaluate_epoch(eval_tuple, train_tuple, iters=-1, epoch=epoch)
            if args.multiGPU:
                torch.distributed.barrier()

            # Save (only on one machine)
            if args.local_rank == 0:
                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    self.save("BEST_EVAL_LOSS")
                self.save("Epoch%02d" % (epoch+1))
            if args.multiGPU:
                torch.distributed.barrier()
            
            # 
            if (args.local_rank == 0) and (args.wandb):
                wandb.log({"loss_valid/avg_eval_loss": avg_eval_loss, "loss_valid/best_eval_loss": best_eval_loss})
            if args.multiGPU:
                torch.distributed.barrier()

    def evaluate(self, valid_batch_size=None, return_cross_relationship_score=False, name="default"):
        topk = -1
        if args.tiny:
            topk=500
        if args.fast:
            topk= 5000

        # Data loader
        train_tuple = get_tuple(args.train, args.batch_size, shuffle=False, drop_last=True, mscoco_only=args.mscoco_only)
        if valid_batch_size is None:
            valid_batch_size = 2048 if args.multiGPU else 512
            valid_batch_size = 256 if args.multiGPU else 256
        eval_tuple = get_tuple(args.valid, valid_batch_size, shuffle=False, drop_last=False, topk=topk, mscoco_only=args.mscoco_only)

        # Eval
        if return_cross_relationship_score:
            avg_eval_loss, cross_relationship_score = self.evaluate_epoch(eval_tuple=eval_tuple, negative_tuple=train_tuple, iters=-1, return_cross_relationship_score=return_cross_relationship_score)
            print("cross_relationship_score:")
            print(cross_relationship_score)
            print(cross_relationship_score.shape)
            path_cmm = f'snap/cross_relationship_score/{name}.txt'
            print(path_cmm)
            np.savetxt(path_cmm, cross_relationship_score, delimiter=',')
            df_cmm = pd.read_csv(path_cmm, header=None)
            df_cmm = df_logit2prob(df_cmm)
            #print(df_cmm.head(5))
            print("df_cmm acc", df_cmm["correct"].mean())
            y_true = df_cmm["is_match"].values
            y_pred =  df_cmm["y_pred"].values
            cfm = confusion_matrix(y_true, y_pred)
            print("="*50, "cfm")
            print(cfm)
            print("="*50)
        else:
            avg_eval_loss = self.evaluate_epoch(eval_tuple=eval_tuple, negative_tuple=train_tuple, iters=-1, return_cross_relationship_score=return_cross_relationship_score)
        print("avg_eval_loss:", avg_eval_loss)
        print("Eval done!")


    def evaluate_epoch(self, eval_tuple: DataTuple, negative_tuple: DataTuple, iters: int=-1, epoch=-1, return_cross_relationship_score=False):
        self.model.eval()
        eval_ld = eval_tuple.loader
        total_loss = 0.
        total_losses = 0.
        uid2ans = {}
        
        if return_cross_relationship_score:
            crs_list = []
            match_list = []
        
        for i, batch in enumerate(tqdm(eval_ld, total=len(eval_ld), desc=f"Eval epoch {epoch}")):
            #for i, batch in enumerate(eval_ld):
            loss, losses, logit, res_dict = self.valid_batch(negative_tuple, batch)
            total_loss += loss
            total_losses += losses

            if return_cross_relationship_score:
                crs_list.append(res_dict["cross_relationship_score"])
                match_list.append(res_dict["matched_labels"])
                
            if args.task_qa:
                score, label = logit.max(1)
                for datum, l in zip(batch, label.cpu().numpy()):
                    uid = datum.uid
                    ans = negative_tuple.dataset.answer_table.id2ans(l)
                    uid2ans[uid] = ans
            if i == iters:
                break

        if return_cross_relationship_score:
            crs_numpy = torch.cat(crs_list, 0).detach().cpu().numpy()
            match_numpy = np.expand_dims(torch.cat(match_list, 0).detach().cpu().numpy(), axis=1)
            crs_numpy = np.concatenate((match_numpy, crs_numpy), axis=1)
        
        if args.report_pi_acc:
            pi_acc  = self.valid_pi_acc.compute()
            pi_accs = self.valid_pi_accs.compute()
        if self.report_qa_acc:
            qa_acc  = self.valid_qa_acc.compute()
            qa_accs = self.valid_qa_accs.compute()
        if self.report_cmm_acc:
            cmm_acc  = self.valid_cmm_acc.compute()
            cmm_cfm  = self.valid_cmm_cfm.compute()
            print("CMM_Acc_valid:", cmm_acc.detach().cpu().numpy().item())
            print("CMM_Cfm_valid:", cmm_cfm.detach().cpu().numpy())
            if args.wandb and (args.local_rank == 0):
                wandb.log({"CMM_Acc_valid": cmm_acc.detach().cpu().numpy().item()})
                wandb.log({"CMM_Cfm_valid": cmm_cfm.detach().cpu().numpy()})
                
        if (args.local_rank == 0) and args.report_pi_acc:
            print("Valid-Epoch", epoch, "PI all,min,mean", pi_accs, pi_accs.min(), pi_accs.mean()) # torch.quantile(pi_accs, torch.tensor(0.05), dim=0, keepdim=False))
            all_accs = np.nan_to_num(pi_accs.detach().cpu().numpy(), nan=1.0)
            print("Valid-Epoch", epoch, "PI all,min,mean (imputed)", all_accs, all_accs.min(), all_accs.mean()) # torch.quantile(pi_accs, torch.tensor(0.05), dim=0, keepdim=False))

            row = np.expand_dims(np.concatenate(
                ([epoch], all_accs),
                axis=0), 0)
            csv_path = os.path.join(args.output, "pi_accs_valid.csv")
            print("Save pi to", csv_path)
            with open(csv_path, 'ab') as f:
                np.savetxt(f, row, fmt="%.4f")
            
            print("")

        if args.wandb and (args.local_rank == 0) and args.report_pi_acc:
            print(pi_acc, pi_accs.std(), pi_accs.min()) # torch.quantile(pi_accs, torch.tensor(0.05), dim=0, keepdim=False))
            wandb.log({"PI_Acc_valid/mean": pi_acc.detach().cpu().numpy().item(), 
                "PI_Acc_valid/std": np.std(np.nan_to_num(pi_accs.detach().cpu().numpy(), nan=1.0)), 
                "PI_Acc_valid/min": np.min(np.nan_to_num(pi_accs.detach().cpu().numpy(), nan=1.0)),
                "PI_Acc_valid/q05": np.quantile(np.nan_to_num(pi_accs.detach().cpu().numpy(), nan=1.0), q=0.05),
                "PI_Acc_valid/q25": np.quantile(np.nan_to_num(pi_accs.detach().cpu().numpy(), nan=1.0), q=0.25)}) 
            #"PI_Acc_train/q05": torch.quantile(pi_accs, torch.tensor(0.05), dim=0, keepdim=False)})

        if args.local_rank == 0:
            print(args.local_rank ,"The valid loss is %0.4f" % (total_loss / len(eval_ld)))
            losses_str = "The losses are "
            for name, loss in zip(LOSSES_NAME, total_losses / len(eval_ld)):
                losses_str += "%s: %0.4f " % (name, loss)
            print(losses_str)

        if args.wandb and args.local_rank == 0:
            wandb.log({"loss/valid": loss, "total_loss/valid": total_loss/len(eval_ld)}, commit=False)
            val_losses_names = [i + "/valid" for i in LOSSES_NAME]
            wandb.log(dict(zip(val_losses_names, total_losses / len(eval_ld))))
        if args.multiGPU:
            torch.distributed.barrier()

        if self.report_pi_acc:
            self.valid_pi_acc.reset()
            self.valid_pi_accs.reset()
        if self.report_qa_acc:
            self.valid_qa_acc.reset()
            self.valid_qa_accs.reset()
        if self.report_cmm_acc:
            self.valid_cmm_acc.reset()
            self.valid_cmm_cfm.reset()
            
        if args.task_qa:
            eval_tuple.evaluator.evaluate(uid2ans, pprint=True)

        if return_cross_relationship_score:
            return total_loss / len(eval_ld), crs_numpy
        else:
            return total_loss / len(eval_ld)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(args.output, "%s_LXRT.pth" % name))

    def load(self, path, strict=True):
        print("Load BERT extractor from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        self.model.load_state_dict(state_dict, strict=strict)

    def load_lxmert(self, path):
        print("Load LXMERT model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Keys in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Keys in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()
        print("Before", self.model.pi_head.conv1x1.bias)
        self.model.load_state_dict(state_dict, strict=False)
        print("After", self.model.pi_head.conv1x1.bias)
        print("Loading done.")

if __name__ == "__main__":
    print("Set Arguments:", args)

    lxmert = LXMERT(max_seq_length=20)
    if args.evaluate:
        lxmert.evaluate()
    elif args.finetune_pi_head:
        print("#### Load weights ####")
        print("#### Freeze ####")
        lxmert.train_pihead_only()
        print("#### Finetune PI head ####")
        lxmert.train() #train on train, and eval on validset
        print("#### Eval PI head ####")

    elif args.evaluate_matchingscore:
        lxmert.evaluate(valid_batch_size=100, return_cross_relationship_score=True, name=f'{args.load_lxmert.split("/")[2]}___{args.valid}')

    elif args.saving_model_name is not None:
        print(f"Load model from {args.load_lxmert}.")
        lxmert.load_lxmert(args.load_lxmert)
        lxmert.model.cpu()
        print(next(lxmert.model.parameters()).device)
        print(f"Save model in {args.saving_model_name}.")
        torch.save(lxmert.model, args.saving_model_name)

    else:
        lxmert.train()

