# coding=utf-8
# Copyleft 2019 project LXRT.

from collections import defaultdict
import json
import random
import re

import numpy as np
from torch.utils.data import Dataset
import torch

from param import args
from pretrain.qa_answer_table import AnswerTable
from utils import load_obj_tsv, load_obj_pkl

import gc

TINY_IMG_NUM = 500
FAST_IMG_NUM = 5000

#----------------------------------------------------------------
def keyword_replacer(x, replace_dict):
    replaced = False
    raw = x
    for keyword in list(replace_dict.keys()):
        if keyword in raw:
            #x = x.replace(keyword, replace_dict[keyword])
            replaced = True
            x = re.sub(r"\b%s\b" % keyword, replace_dict[keyword], x)
    return x, replaced

replace_dict = { #"X"
"left": "right", "right": "left",
 # Y
"above": "below", "below": "above",
"under": "over", "over": "under",
# Z
 "foreground": "background", "background": "foreground",
"in front of": "behind", "behind": "in front of",
#"back": "front", "front":"back"
}
#----------------------------------------------------------------


Split2ImgFeatPath = {
    'mscoco_train': 'data/mscoco_imgfeat/train2014_obj36.tsv',
    'mscoco_minival': 'data/mscoco_imgfeat/val2014_obj36.tsv',
    'mscoco_nominival': 'data/mscoco_imgfeat/val2014_obj36.tsv',
    'mscoco_minival_withpi': 'data/mscoco_imgfeat/val2014_obj36.tsv', # for cross-rel-score analysis for samples with pi in it
    'mscoco_minival_withpermutedpi': 'data/mscoco_imgfeat/val2014_obj36.tsv', # for cross-rel-score analysis for samples with PERMUTED pi in it
        'mscoco_minival_withpi2': 'data/mscoco_imgfeat/val2014_obj36.tsv', # for cross-rel-score analysis for samples with pi in it
        'mscoco_minival_withpermutedpi2': 'data/mscoco_imgfeat/val2014_obj36.tsv', # for cross-rel-score analysis for samples with PERMUTED pi in it
    'vgnococo': 'data/vg_gqa_imgfeat/vg_gqa_obj36.tsv',
}


class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self, uid, sent, visual_feats=None,
                 obj_labels=None, attr_labels=None,
                 is_matched=None, label=None, pi_labels=None):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats # feat, boxes/objs_pos 
        self.obj_labels = obj_labels
        self.attr_labels = attr_labels
        self.is_matched = is_matched  # whether the visual and obj matched
        self.label = label
        self.pi_labels = pi_labels


class LXMERTDataset:
    def __init__(self, splits: str, qa_sets=None):
        """
        :param splits: The data sources to be loaded
        :param qa_sets: if None, no action
                        o.w., only takes the answers appearing in these dsets
                              and remove all unlabeled data (MSCOCO captions)
        """
        self.name = splits
        self.sources = splits.split(',')

        # Loading datasets to data
        self.data = []
        for source in self.sources:
            self.data.extend(json.load(open("data/lxmert/%s.json" % source)))
        print("Load %d data from %s" % (len(self.data), self.name))

        # Create answer table according to the qa_sets
        self.answer_table = AnswerTable(qa_sets)
        print("Load an answer table of size %d." % (len(self.answer_table.ans2id_map())))

        # Modify the answers
        for datum in self.data:
            labelf = datum['labelf']
            for cat, labels in labelf.items():
                for label in labels:
                    for ans in list(label.keys()):
                        new_ans = self.answer_table.convert_ans(ans)
                        if self.answer_table.used(new_ans):
                            if ans != new_ans:
                                label[new_ans] = label.pop(ans)
                        else:
                            label.pop(ans)

    def __len__(self):
        return len(self.data)


def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx),


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features",
              depth_mid, depth_min, depth_max, depth_mea, depth_med, pi_labels]
"""
class LXMERTTorchDataset(Dataset):
    def __init__(self, dataset: LXMERTDataset, use_pkl=None, center_only=False, 
            depth_type=None, area=None, topk=-1, mscoco_only=False, matching_prob=.5, task_pi_cl_cmm=False):
        super().__init__()
        self.raw_dataset = dataset
        self.task_matched = args.task_matched

        self.use_pkl = use_pkl
        self.center_only = center_only
        self.depth_type = depth_type
        self.area = area
        self.mscoco_only = mscoco_only
        self.matching_prob = matching_prob
        self.task_pi_cl_cmm = task_pi_cl_cmm
        
        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM

        # Load the dataset
        img_data = []
        for source in self.raw_dataset.sources:
            if self.depth_type or self.area or self.use_pkl:
                # rm .tsv and conc new postfix
                fname = Split2ImgFeatPath[source][:-4] + "_depth_v3.pkl"
                print(f"Loading {fname}.")
                pkl_data = load_obj_pkl(fname)
                print(f"Loading {fname} done.")
                if topk:
                    pkl_data = pkl_data[:topk]

                img_data.extend(pkl_data)
                del pkl_data
                gc.collect()
            else:
                img_data.extend(load_obj_tsv(Split2ImgFeatPath[source], topk))

        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Filter out the dataset
        used_data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                used_data.append(datum)

        # Flatten the dataset (into one sent + one image entries)
        self.data = []
        for datum in used_data:
            sentf = datum['sentf']
            
            if self.mscoco_only:
                tmp = sentf['mscoco2']
                sentf.clear()
                sentf['mscoco2'] = tmp 

            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:
                    labels = datum['labelf'][sents_cat]
                else:
                    labels = None
                for sent_idx, sent in enumerate(sents):
                    new_datum = {
                        'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                        'img_id': datum['img_id'],
                        'sent': sent
                    }
                    if labels is not None:
                        new_datum['label'] = labels[sent_idx]
                    self.data.append(new_datum)
        print(f"Use {len(self.data)} data in torch dataset for rank {args.local_rank}.")

    def __len__(self):
        return len(self.data)

    def random_feat(self):
        """Get a random obj feat from the dataset."""
        datum = self.data[random.randint(0, len(self.data)-1)]
        img_id = datum['img_id']
        img_info = self.imgid2img[img_id]
        feat = img_info['features'][random.randint(0, 35)]
        return feat

    def __getitem__(self, item: int):
        datum = self.data[item]

        uid = datum['uid']
        img_id = datum['img_id']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        objs_pos = img_info['boxes'].copy() # renaming: boxes to objs_pos(ition)
        obj_labels = img_info['objects_id'].copy()
        obj_confs = img_info['objects_conf'].copy()
        attr_labels = img_info['attrs_id'].copy()
        attr_confs = img_info['attrs_conf'].copy()
        assert obj_num == len(objs_pos) == len(feats)

        # Normalize the objs_pos (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        objs_pos = img_info['boxes'].copy()

        objs_pos_real = img_info['boxes'].copy()
        objs_pos_real[:, (0, 2)] /= img_w
        objs_pos_real[:, (1, 3)] /= img_h
    
     
        
        np.testing.assert_array_less(objs_pos_real, 1+1e-5)
        np.testing.assert_array_less(-objs_pos_real, 0+1e-5)
        pi_labels = img_info['PI_labels'].copy()

        #
        #
        #
        #============================================================
        # lxmerdt
        #============================================================
        if args.old:    
            # calc area
            if self.area:
                objs_area = (objs_pos[:, 2] - objs_pos[:, 0]) * (objs_pos[:, 3] - objs_pos[:, 1])
            
            # object depth value to objs_pos 
            if self.center_only:
                objs_pos = img_info['center_bxs'].copy().astype("float32")
                objs_pos[:, 0] /= img_w
                objs_pos[:, 1] /= img_h
            
            if args.real_center_only:
                objs_pos = np.zeros(shape=img_info['center_bxs'].shape).astype("float32")
                objs_bb = img_info['boxes'].copy().astype("float32")
                objs_pos[:, 0] = (objs_bb[:, 0] +  0.5 * (objs_bb[:, 2] - objs_bb[:, 0]) ) /  img_w
                objs_pos[:, 1] = (objs_bb[:, 1] +  0.5 * (objs_bb[:, 2] - objs_bb[:, 1]) ) /  img_h
    
            
            # add object area values
            if self.area:
                #objs_area = img_info['area'].copy()
                objs_area = objs_area[:,np.newaxis]
                objs_pos = np.concatenate([objs_pos, objs_area], axis=1).astype("float32")
            
            # add depth type
            if self.depth_type:
                objs_depth = img_info[self.depth_type].copy()
                
                if args.quant:  
                    bins = np.array([0.0, .333, .666])
                    objs_depth = np.digitize(objs_depth, bins) / 3
                    objs_depth = objs_depth.astype("float32")
                objs_depth = objs_depth[:,np.newaxis]
                objs_pos = np.concatenate([objs_pos, objs_depth], axis=1).astype("float32")
            
            # add depth std
            if args.add_std:
                objs_depthstd = img_info["depth_std"].copy()
                objs_depthstd = objs_depthstd[:,np.newaxis]
                objs_pos = np.concatenate([objs_pos, objs_depthstd], axis=1).astype("float32")
            
            # add depth q25,q75
            if args.add_quantiles:
                objs_depthq25 = img_info["depth_q25"].copy()[:,np.newaxis]
                objs_depthq75 = img_info["depth_q75"].copy()[:,np.newaxis]
                objs_pos = np.concatenate([objs_pos, objs_depthq25, objs_depthq75], axis=1).astype("float32")
            
            if args.nopi:
                objs_pos_shape = objs_pos.shape
                objs_pos = np.zeros(objs_pos_shape).astype("float32")
            
            # task_pi_aux
            pi_labels = img_info['PI_labels'].copy()

        #============================================================
        #         
        # x_center
        # y_center,
        # x1,
        # y1,
        # x2,
        # y2,
        # a_rel
        # a_abs,
        # w
        # h
        # d_med
        # d_mean
        # d_center
        # d_std
        # d_q25
        # d_q75
        #============================================================
        if args.new:
            bs = feats.shape[0]
            PI = np.zeros((bs, 20))
            if args.use_center:
                tmp = img_info['center_bxs'].copy().astype("float32")
                tmp[:, 0] /= img_w
                tmp[:, 1] /= img_h
                PI[:, :2] = tmp

            if args.use_bb:
                tmp = objs_pos.copy().astype("float32")
                tmp[:, (0, 2)] /= img_w
                tmp[:, (1, 3)] /= img_h
                PI[:, 2:6] = tmp

            if args.use_area_rel:
                tmp = img_info['boxes'].copy()[:,np.newaxis].astype("float32")
                tmp[:, (0, 2)] /= img_w
                tmp[:, (1, 3)] /= img_h
                PI[:, 6:7] = (tmp[:, 2] - tmp[:, 0]) * (tmp[:, 3] - tmp[:, 1])
                
            if args.use_area_absolute:
                tmp = img_info['boxes'].copy().astype("float32")
                PI[:, 7:8] = (tmp[:, 2] - tmp[:, 0]) * (tmp[:, 3] - tmp[:, 1])

            if args.use_wh:
                tmp = np.array([[img_w, img_h]]* bs).astype("float32") 
                PI[:, 8:10] = tmp

            if args.use_d_med:
                PI[:, 10:11] = img_info["depth_med"].copy()[:,np.newaxis].astype("float32") 
            if args.use_d_mean:
                PI[:, 11:12] = img_info["depth_mea"].copy()[:,np.newaxis].astype("float32") 
            if args.use_d_cntr:
                PI[:, 12:13] = img_info["depth_mid"].copy()[:,np.newaxis].astype("float32") 
            if args.use_d_std:
                PI[:, 13:14] = img_info["depth_std"].copy()[:,np.newaxis].astype("float32") 
            if args.use_d_q25:
                PI[:, 14:15] = img_info["depth_q25"].copy()[:,np.newaxis].astype("float32") 
            if args.use_d_q75:
                PI[:, 15:16] = img_info["depth_q75"].copy()[:,np.newaxis].astype("float32") 
            if args.use_d_quant:
                tmp = img_info["depth_med"].copy()
                bins = np.array([0.0, .333, .666])
                tmp = np.digitize(tmp, bins) / 3
                tmp = tmp[:,np.newaxis].astype("float32")
                PI[:, 16:17]
            objs_pos = PI.astype("float32")
        # /end lxmerdt
        #========================================================

        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 1
        sent = datum['sent']
        sent_raw = datum['sent']
        #========================================================
        is_with_pi = False
        if self.task_pi_cl_cmm:
            sent_new, is_with_pi = keyword_replacer(sent, replace_dict)
            if is_with_pi:
                if random.random() < 0.5:
                    sent = sent_new
                    is_matched = 0
        #========================================================
        
        if self.task_matched and not is_with_pi:
            # matching_prob=0.5 -> 50/50, matching_prob=-1 -> only_matches, matching_prob=1.1 -> no_matches 
            if random.random() < self.matching_prob:
                is_matched = 0
                other_datum = self.data[random.randint(0, len(self.data)-1)]
                while other_datum['img_id'] == img_id:
                    other_datum = self.data[random.randint(0, len(self.data)-1)]
                sent = other_datum['sent']

        # Label, convert answer to id
        if 'label' in datum:
            label = datum['label'].copy()
            for ans in list(label.keys()):
                label[self.raw_dataset.answer_table.ans2id(ans)] = label.pop(ans)
        else:
            label = None

        # Create target
        example = InputExample(
            uid, sent, (feats, objs_pos),
            (obj_labels, obj_confs), (attr_labels, attr_confs),
            is_matched, label, pi_labels
        )
        return example


class LXMERTEvaluator:
    def __init__(self, dataset: LXMERTDataset):
        self.raw_dataset = dataset

        # Create QA Eval Data
        self.data = []
        for datum in self.raw_dataset.data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:    # A labeled dataset
                    labels = datum['labelf'][sents_cat]
                    for sent_idx, sent in enumerate(sents):
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent': sent,
                            'dset': sents_cat,
                            'label': labels[sent_idx]
                        }
                        self.data.append(new_datum)

        # uid2datum
        self.uid2datum = {}
        for datum in self.data:
            self.uid2datum[datum['uid']] = datum

    def evaluate(self, uid2ans: dict, pprint=False):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        accu = score / cnt
        dset2accu = {}
        for dset in dset2cnt:
            dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

        if pprint:
            accu_str = "Overall Accu %0.4f, " % (accu)
            sorted_keys = sorted(dset2accu.keys())
            for key in sorted_keys:
                accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
            print(accu_str)

        return accu, dset2accu

    def dump_result(self, uid2ans: dict, path):
        raise NotImplemented
