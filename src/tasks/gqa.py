# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator
from pathlib import Path
from datetime import datetime

import wandb
from evaluation import make_evaluation
from hurry.filesize import size

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    tset = GQATorchDataset(dset, args.use_pkl, args.center_only, args.real_center_only, args.depth_type, args.area)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class GQA:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            valid_bsize=32
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        if args.test != "":
            test_bsize = 2048 if args.multiGPU else 512
            test_bsize = 32
            self.test_tuple = get_tuple(
                args.test, bs=test_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.test_tuple = None

        self.model = GQAModel(self.train_tuple.dataset.num_answers,
            visual_pos_dim=args.visual_pos_dim, gqa_dropout_rate=args.gqa_dropout_rate)
        print(self.model)
        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            qa_with_dropout = args.gqa_dropout_rate > 0.0
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans, qa_with_dropout=qa_with_dropout)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple, test_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        best_test = 0.
        valid_improvement=False
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                if args.mce_loss:
                    max_value, target = target.max(1)
                    loss = self.mce_loss(logit, target) * logit.size(1)
                else:
                    loss = self.bce_loss(logit, target)
                    loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans
                if args.wandb:
                    wandb.log({"Stat/epochs": epoch, "Stat/abs_steps": i})
                    wandb.log({"training/loss": loss})

            train_score = evaluator.evaluate(quesid2ans) 
            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, train_score * 100.)
            self.save(f"EPOCH_{epoch}")

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    valid_improvement=True
                    self.save("BEST")
                log_str += "Epoch %d: Valid %0.2f\t" % (epoch, valid_score * 100.) + "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)
            
            if self.test_tuple is not None:  # Do Test
                test_score = self.evaluate(test_tuple)
                if valid_improvement: # valid not test!
                    best_test = test_score
                    valid_improvement=False
                log_str += "Epoch %d: Test %0.2f\t" % (epoch, test_score * 100.) + "Epoch %d: Best %0.2f\n" % (epoch, best_test * 100.)
     
            print(log_str, end='')

            if args.wandb:
                wandb.log({"loss/train_score": train_score * 100., "loss/valid_score": valid_score * 100., "loss/best_valid": best_valid*100., "loss/test_score": test_score*100., "loss/best_test": best_test*100.})
                    
            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        #self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None, topk=1):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(tqdm(loader)):
            #print("datum_tuple:", len(datum_tuple))
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                #import pdb; pdb.set_trace()
                if topk == 1:
                    score, label = logit.max(1)
                    for qid, l in zip(ques_id, label.cpu().numpy()):
                        ans = dset.label2ans[l]
                        quesid2ans[qid] = ans
                else:
                    score_topk, label_topk = torch.topk(logit, topk)
                    for qid, ls in zip(ques_id, label_topk.cpu().numpy()):
                        ans_topk = []
                        for l in ls:
                            ans_topk.append(dset.label2ans[l])
                        quesid2ans[qid] = ans_topk
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump=dump)
        return evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            if i %2==0:
                print(i, end = "")
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    if args.wandb:
        wandb.init(project=f"lxmerdt_gqa2")
        config = wandb.config
        config.lxmerdt_name = args.sname
        print(f"WANDB with config.lxmerdt_name={config.lxmerdt_name}")
        config.host_name = args.host_name
        config.dgxname = args.dgxname
        config.tiny = args.tiny
        config.train_data = args.train
        config.valid_data = args.valid
        config.bs = args.batch_size
        config.lr = args.lr
        config.seed = args.seed
        config.gqa_dropout_rate = args.gqa_dropout_rate

        config.depth_permute = args.depth_permute
        config.depth_randomize = args.depth_randomize
        config.all_permute = args.all_permute
        config.all_randomize = args.all_randomize

   
    # Build Class
    gqa = GQA()
    print("GQA model sucessfully created.")

    # Load Model
    if args.load is not None:
        gqa.load(args.load)

    # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
    print('Splits in Train data:', gqa.train_tuple.dataset.splits)
    print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
    print('Splits in Test data:', gqa.test_tuple.dataset.splits)
    print("_")
    
    if not args.evaluate:
        print("TRAIN", datetime.now())
        gqa.train(gqa.train_tuple, gqa.valid_tuple, gqa.test_tuple)
    else:
        print("SKIP TRAINING", datetime.now())

    # Load model with best valid result
    print("Name:", args.sname, datetime.now())
    gqa.load(f"snap/gqa2/{args.sname}/BEST")

    print("Evaluate for testdev on Top1", datetime.now())
    result = gqa.evaluate(gqa.test_tuple, dump=os.path.join(args.output, 'testdev_predict.json'))
    print(result)
    
    print(f"Evaluate for testdev on Top5", datetime.now())
    dump_path = os.path.join(args.output, f'testdev_predict_top5.json')
    gqa.predict(gqa.test_tuple, dump=dump_path, topk=5)

    print(f"Evaluate for testdev on Top10", datetime.now())
    dump_path = os.path.join(args.output, f'testdev_predict_top10.json')
    gqa.predict(gqa.test_tuple, dump=dump_path, topk=10)


    print("GC")
    gqa.model.cpu()
    del gqa.model
    del gqa
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    #print(f"Dumped in {dump_path}.")
    print("---MAKE EVALUATION---", datetime.now())

    print("Start")
    res = make_evaluation(f"{args.sname}")

    if args.wandb:
        config.RES_Top1_Acc = res["r1"]["res_N"]["Acc"]
        config.RES_Top5_Acc = res["r5"]["res_N"]["Acc"]
        config.RES_Top1_x = res["r1"]["res_x"]["Acc"]
        config.RES_Top1_y = res["r1"]["res_y"]["Acc"]
        config.RES_Top1_z = res["r1"]["res_z"]["Acc"]
        config.RES_Top1_p = res["r1"]["res_p50"]["Acc"]
    try:
        print(res)
        print("---")
        print("Top1", res["r1"]["res_N"]["Acc"])
        print("X", res["r1"]["res_x"]["Acc"])
        print("Y", res["r1"]["res_y"]["Acc"])
        print("Z", res["r1"]["res_z"]["Acc"])
        print("P", res["r1"]["res_p50"]["Acc"])
        print("Top5", res["r5"]["res_N"]["Acc"])
    except:
        print("res not printable")

    print("GQA DONE", datetime.now())
