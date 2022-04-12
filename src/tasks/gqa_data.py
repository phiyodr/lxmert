# coding=utf-8
# Copyleft 2019 project LXRT.

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv, load_obj_pkl

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000


class GQADataset:
    """
    A GQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/gqa/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/gqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/gqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class GQABufferLoader():
    def __init__(self):
        self.key2data = {}

    def load_data(self, name, number):
        if name == 'testdev':
            path = "data/vg_gqa_imgfeat/gqa_testdev_obj36.tsv"
        else:
            path = "data/vg_gqa_imgfeat/vg_gqa_obj36.tsv"
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(
                path,
                topk=number
            )
        return self.key2data[key]


gqa_buffer_loader = GQABufferLoader()


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class GQATorchDataset(Dataset):
    def __init__(self, dataset: GQADataset, use_pkl=None, center_only=False, 
            real_center_only=False, depth_type=None, area=None):
        super().__init__()
        self.raw_dataset = dataset

        #==========================================================
        # lxmerdt
        self.use_pkl = use_pkl
        self.center_only = center_only
        self.depth_type = depth_type        
        self.real_center_only = real_center_only
        self.area = area
        print("---In GQATorchDataset", self.depth_type)
        # /lxmerdt
        #==========================================================

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []

        #==========================================================
        # lxmerdt

        if self.depth_type or self.area or self.use_pkl:
            if 'testdev' in dataset.splits or 'testdev_all' in dataset.splits:
                path = "data/vg_gqa_imgfeat/gqa_testdev_obj36_depth.pkl"    
                path = "data/vg_gqa_imgfeat/gqa_testdev_obj36_depth_v2.pkl"    
                img_data.extend(load_obj_pkl(path))
                print(f"Loaded {path}.")
            else:
                path = "data/vg_gqa_imgfeat/vg_gqa_obj36_depth.pkl"
                path = "data/vg_gqa_imgfeat/vg_gqa_obj36_depth_v2.pkl"
                img_data.extend(load_obj_pkl(path))  
                print(f"Loaded {path}.")
                if topk:
                    img_data = img_data[:topk]
                    img_data = img_data.copy()    

                print(f"Length img_data: {len(img_data)} (topk={topk})")      
        #
        else:
            if 'testdev' in dataset.splits or 'testdev_all' in dataset.splits:     # Always loading all the data in testdev
                img_data.extend(gqa_buffer_loader.load_data('testdev', -1))
            else:
                img_data.extend(gqa_buffer_loader.load_data('train', topk))
        
        # /lxmerdt
        #==========================================================
        
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        objs_pos = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(objs_pos) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        objs_pos = objs_pos.copy()
        objs_pos[:, (0, 2)] /= img_w
        objs_pos[:, (1, 3)] /= img_h
        np.testing.assert_array_less(objs_pos, 1+1e-5)
        np.testing.assert_array_less(-objs_pos, 0+1e-5)

    
        #============================================================
        # lxmerdt old
        #import pdb; pdb.set_trace()
        if args.old:
            # calc area
            if self.area:
                objs_area = (objs_pos[:, 2]-objs_pos[:, 0]) * (objs_pos[:, 3]-objs_pos[:, 1])
                #objs_area = objs_area[:,np.newaxis]

            # object depth value to objs_pos 
            if self.center_only:
                objs_pos = img_info['center_bxs'].copy().astype("float32")
                objs_pos[:, 0] /= img_w
                objs_pos[:, 1] /= img_h

            if args.real_center_only:
                objs_pos0 = img_info['boxes'].copy()
                objs_pos  = img_info['boxes'].copy()

                objs_pos[:, 0] = (objs_pos0[:, 0] +  0.5 * (objs_pos0[:, 2] - objs_pos0[:, 0]) ) /  img_w
                objs_pos[:, 1] = (objs_pos0[:, 1] +  0.5 * (objs_pos0[:, 2] - objs_pos0[:, 1]) ) /  img_h
                objs_pos = objs_pos[:, :2] 

            # add object area values
            if self.area:
                #objs_area = img_info['area'].copy()
                objs_area = objs_area[:,np.newaxis]
                objs_pos = np.concatenate([objs_pos, objs_area], axis=1).astype("float32")
                #print("__getitem__1", objs_pos.shape, objs_pos.dtype)

            # add depth type
            if self.depth_type:
                #print("------indepththingy")
                objs_depth = img_info[self.depth_type].copy()

                if args.quant:  
                    bins = np.array([0.0, .333, .666])
                    objs_depth = np.digitize(objs_depth, bins) / 3
                    objs_depth = objs_depth.astype("float32")

                #============================================================
                # Randomize/permute testing

                if args.depth_zero:
                    objs_depth = np.zeros((objs_depth.shape[0],), dtype="float32")
                if args.depth_randiter:
                    objs_depth = np.random.rand(objs_depth.shape[0]).astype("float32")
                    objs_depth = np.where(objs_depth > 0.5, 1., 0.).astype("float32")
                if args.depth_iter:
                    interim = [0,1] * int(objs_depth.shape[0]/2)
                    objs_depth = np.array(interim).astype("float32")
                if args.depth_permute:
                    objs_depth = np.random.permutation(objs_depth)
                if args.depth_randomize:
                    objs_depth = np.random.rand(objs_depth.shape[0]).astype("float32")
                # /end ranomize/permute testing 
                #============================================================

                objs_depth = objs_depth[:,np.newaxis]
                #print("==2>", objs_depth.shape)
                objs_pos = np.concatenate([objs_pos, objs_depth], axis=1).astype("float32")
                #print("==3>", objs_pos.shape, objs_pos.dtype)
            else:
                pass
                #print("Not in depththingy", self.depth_type)

            # add depth std
            if args.add_std:
                objs_depthstd = img_info["depth_std"].copy()
                objs_depthstd = objs_depthstd[:,np.newaxis]
                #print("==2>", objs_depth.shape)
                objs_pos = np.concatenate([objs_pos, objs_depthstd], axis=1).astype("float32")

            # add depth q25,q75
            if args.add_quantiles:
                objs_depthq25 = img_info["depth_q25"].copy()[:,np.newaxis]
                objs_depthq75 = img_info["depth_q75"].copy()[:,np.newaxis]
                objs_pos = np.concatenate([objs_pos, objs_depthq25, objs_depthq75], axis=1).astype("float32")

        # /end lxmerdt old
        #============================================================
        # lxmerdt new
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
                PI[:, 16:17] = tmp
            objs_pos = PI.astype("float32")
        #============================================================
        # Randomize/permute testing

        if args.all_zero:
            objs_pos = np.zeros((objs_pos.shape[0], objs_pos.shape[1]), dtype="float32")
        if args.all_randiter:
            objs_pos = np.random.rand(objs_pos.shape[0],objs_pos.shape[1]).astype("float32")
            objs_pos = np.where(objs_pos > 0.5, 1., 0.).astype("float32")
        if args.all_iter:
            raise NotImplemented 
            #interim = [0,1] * int(objs_depth.shape[0]/2)
            #objs_depth = np.array(interim).astype("float32")        
        if args.all_permute:
            objs_pos = np.random.permutation(objs_pos)
        if args.all_randomize:
            objs_pos = np.random.rand(objs_pos.shape[0],objs_pos.shape[1]).astype("float32")

        # /end Randomize/permute testing
        #============================================================

        # Create target
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, objs_pos, ques, target
        else:
            return ques_id, feats, objs_pos, ques


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }

        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


