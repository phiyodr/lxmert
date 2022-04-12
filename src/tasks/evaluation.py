# eval perf on gqa with subsets on XYZ
import os, sys
import pandas as pd
import re
from pathlib import Path
import json
import numpy as np
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
candidate_label = ["position"]

import torch

#https://stackoverflow.com/a/57915246
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def any_intersection(sent, word_list):
    return any(x in sent.split(" ") for x in word_list)

def clean(sent):
    regex_sub = re.sub(r"[',.;@#?!&$]+", ' ', sent)  # + means match one or more
    return re.sub(r"\s+", ' ', regex_sub)  # Replaces any number of spaces with one space
 
def get_correct(df, target):
    if target:
        n_total = df[df[target] == True].shape[0]
        n_correct = df[df[target] == True]["pred_correct"].sum()
    else:
        n_total = df.shape[0]
        n_correct = df["pred_correct"].sum()
    return {"name":target, "Acc": n_correct/n_total, "N_cor":n_correct, "N_total":n_total}

def get_results(rawdata, results, top1=True):
    data = pd.DataFrame(rawdata)
    res = pd.DataFrame(results)
    df = data.merge(res, left_on='question_id', right_on='questionId')

    del df['question_id']
    del df['questionId']
    del df['img_id']

    df['label_clean'] = df['label'].apply(lambda x: list(x.keys())[0])
    if top1:
        df["pred_correct"] = df.apply(lambda x: x["label_clean"] == x["prediction"], axis=1) 
    else:
        df["pred_correct"] = df.apply(lambda x: x["label_clean"] in x["prediction"], axis=1) 

    print("GR: Lambdas")
    df["has_o"] = df["sent"].apply(lambda x: any_intersection(clean(x), o_words))
    df["has_w"] = df["sent"].apply(lambda x: any_intersection(clean(x), ["where"]))
    df["has_x"] = df["sent"].apply(lambda x: any_intersection(clean(x), x_words))
    df["has_y"] = df["sent"].apply(lambda x: any_intersection(clean(x), y_words))
    df["has_z"] = df["sent"].apply(lambda x: any_intersection(clean(x), z_words))
    sents = df["sent"].tolist()
    #import pdb; pdb.set_trace()
    print("GR: p classifier:")
    len_sents = len(sents)
    classif_results = []
    for (von, bis) in [(0,2500), (2500,5000), (5000,7500), (7500,10000), (10000,len_sents)]:
        print(von,bis)
        classif_result = classifier(sents[von:bis], candidate_label)
        classif_results.extend(classif_result)
    del classif_result
    print("GR: done")
    #import pdb; pdb.set_trace()
    try:    
        print("x0 version")
        df["has_p50"] = [x[0]['scores'][0] > .5 for x in classif_results]
    except:
        print("x version")
        df["has_p50"] = [x['scores'][0] > .5 for x in classif_results]
    
    #print("Preprocessing done.")

    res_N = get_correct(df, None)
    res_o = get_correct(df, "has_o")
    res_w = get_correct(df, "has_w")
    res_x = get_correct(df, "has_x")
    res_y = get_correct(df, "has_y")
    res_z = get_correct(df, "has_z")
    res_p50 = get_correct(df, "has_p50")
    print("GR: getcorr done")
    return {"res_N": res_N, "res_o": res_o, "res_w": res_w, "res_x": res_x, "res_y": res_y, "res_z": res_z, "res_p50": res_p50}

o_words = [ "nearby", "next to", "corner", "close", "neighboring", "near"]
x_words = ["left", "right", "beside", "besides", "alongside", "side" ]
y_words = ["top", "down", "above", "below", "under", "beneath", "underneath", "over", "beyond", "overhead"]
z_words = ["behind", "front", "rear", "back", "ahead", "foreground", "background", "before", "forepart", "far end", "hindquarters"]

def make_evaluation(lxmerdt_name, lxdt_path=None):
    print("MAKE EVAL")
    

    tt = torch.cuda.get_device_properties(0).total_memory
    rr = torch.cuda.memory_reserved(0)
    aa = torch.cuda.memory_allocated(0)
    ff = rr-aa  # free inside reserved


    if lxdt_path is None:
        lxdt_path = Path('/root/plxmert')
    print("lxdt_path=", lxdt_path)

    # rawdata
    print("Load rawdata")
    rawdata_path = lxdt_path / "data/gqa/testdev.json"
    with open(rawdata_path) as json_file:
        rawdata_testdev = json.load(json_file)

    # results
    print("Load results")
    data_path = lxdt_path / f"snap/gqa/{lxmerdt_name}/testdev_predict.json"
    with open(data_path) as json_file:
        res_testdev = json.load(json_file)

    data_path = lxdt_path / f"snap/gqa/{lxmerdt_name}/testdev_predict_top5.json"
    with open(data_path) as json_file:
        res_testdev_top5 = json.load(json_file)
        
    data_path = lxdt_path / f"snap/gqa/{lxmerdt_name}/testdev_predict_top10.json"
    with open(data_path) as json_file:
        res_testdev_top10 = json.load(json_file)

    print("Calc results")
    r1 = get_results(rawdata=rawdata_testdev, results=res_testdev)
    print("Top1 done.", end=" ")
    r5 = get_results(rawdata=rawdata_testdev, results=res_testdev_top5, top1=False)
    print("Top5 done.", end=" ")
    r10 =get_results(rawdata=rawdata_testdev, results=res_testdev_top10, top1=False)
    print("Top10 done.")

    print("TestDev Top1")
    print(r1)
    print("="*50)
    print("TestDev Top5")
    print(r5)
    print("="*50)
    print("TestDev Top10")
    print(r10)
    print("="*50)
    result_path = lxdt_path / f"snap/gqa/{lxmerdt_name}/accs.jsonl"
    with open(result_path, "w") as outfile:
        json.dump(r1, outfile, cls=NpEncoder)
        json.dump(r5, outfile, cls=NpEncoder)
        json.dump(r10, outfile, cls=NpEncoder)

    print("Done!")
    return {"r1": r1, "r5": r5, "r10": r10}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lxmerdt_name', type=str)
    parser.add_argument('--lxdt_path', type=str, default=None)
    args = parser.parse_args()

    print("Start evaluation")
    make_evaluation(args.lxmerdt_name, args.lxdt_path)
    print("===")
