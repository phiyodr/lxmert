#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import scipy
from tqdm import tqdm
import argparse

#input_file = "gqa_testdev_obj36_depth_v2.pkl"
### 1. center_smaller

def is_smaller(obj0, obj1):
    return obj0 < obj1

def compare_every_value_with_every_value(func, vec0, vec1=None):
    if vec1 is None:
        vec1 = vec0
    dim = len(vec0)
    res = np.zeros((dim, dim))
    for i, obj0 in enumerate(vec0):
        for j, obj1 in enumerate(vec1):
            res[i, j] = func(obj0, obj1)
    return res

### ### 2. is_complete_inside_array

def is_complete_inside(obj0, obj1):
    obj0_x1, obj0_y1, obj0_x2, obj0_y2 = obj0
    obj1_x1, obj1_y1, obj1_x2, obj1_y2 = obj1
    if (obj0_x1 > obj1_x1) and (obj0_x2 < obj1_x2) and (obj0_y1 > obj1_y1) and (obj0_y2 < obj1_y2):
        return 1.
    else:
        return 0.

def is_complete_inside_array(arr):
    length = arr.shape[0]
    res = np.zeros((length, length))
    for i, row0 in enumerate(arr):
        for j, row1 in enumerate(arr):
            r = is_complete_inside(row0, row1)
            res[i, j] = r
            #print("i=", i, "j=", j, "|", r, row0, row1)
    return res

### 3. is_overlapping_array

def is_overlapping(obj0, obj1):
    """
    https://stackoverflow.com/a/42874377
    Calculate the Intersection over Union (IoU) of two bounding boxes. Use this to determine if objs are overlapping.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    bb1 = dict()
    bb1['x1'], bb1['y1'], bb1['x2'], bb1['y2'] = obj0

    bb2 = dict()
    bb2['x1'], bb2['y1'], bb2['x2'], bb2['y2'] = obj1

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    if iou > 0.:
        return 1.
    else:
        return 0.

def is_overlapping_array(arr):
    length = arr.shape[0]
    res = np.zeros((length, length))
    for i, row0 in enumerate(arr):
        for j, row1 in enumerate(arr):
            r = is_overlapping(row0, row1)
            res[i, j] = r
            #print("i=", i, "j=", j, "|", r, row0, row1)
    return res

### 4./5. is_inbetween/is_med_in_qs

def is_inbetween(med, q25, q75):
    if (med >= q25) and (med <= q75):
        return 1.
    else:
        return 0.

def is_med_in_qs_array(vec0, vec1, vec2):
    if vec1 is None:
        vec1 = vec0
    length = len(vec0)
    res = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            res[i, j] = is_inbetween(vec0[i], vec1[j], vec2[j])
    return res

def count_approx_pixels(box):
    x1,y1,x2,y2 = box
    area = (x2-x1) * (y2-y1)
    return int(area)


### 6. sig_in_front

from scipy.stats import ttest_ind_from_stats
def ttest_array(means, stds, boxes, alternative="less", pval=.05):
    length = len(means)
    res = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            res[i, j] = ttest_ind_from_stats(mean1=means[i], std1=stds[i], nobs1=count_approx_pixels(boxes[i]),
                     mean2=means[j], std2=stds[j], nobs2=count_approx_pixels(boxes[j]), alternative=alternative).pvalue < pval
    return res


### Main

def generate_PI_labels(boxes, centers, depth_med, depth_mea, depth_std, depth_q25, depth_q75):
    
    x_center_smaller = compare_every_value_with_every_value(is_smaller, vec0=centers[:, 0])
    y_center_smaller = compare_every_value_with_every_value(is_smaller, vec0=centers[:, 1])

    x_completely_left = compare_every_value_with_every_value(is_smaller, vec0=boxes[:, 2], vec1=boxes[:, 0])
    y_completely_top = compare_every_value_with_every_value(is_smaller, vec0=boxes[:, 3], vec1=boxes[:, 1])

    xy_completely_inside = is_complete_inside_array(boxes)
    xy_somehow_overlapping = is_overlapping_array(boxes)
    
    z_median_moreforeground = compare_every_value_with_every_value(is_smaller, vec0=depth_med)
    z_median_bet_qs = is_med_in_qs_array(vec0=depth_med, vec1=depth_q25, vec2=depth_q75)
    z_sigdiff = ttest_array(depth_mea, depth_std, boxes, alternative="less")
    
    return np.stack([x_center_smaller, y_center_smaller, 
              x_completely_left, y_completely_top, 
              xy_completely_inside, xy_somehow_overlapping, 
              z_median_moreforeground, z_median_bet_qs, z_sigdiff], axis=2)

categories = ["x_center_smaller","y_center_smaller","x_completely_left","y_completely_top","xy_completely_inside","xy_somehow_overlapping","z_median_moreforeground","z_median_bet_qs","z_sigdiff"]


def add_pi_labels_to_pkl(input_file, replace_str = ("v2", "v3")):
    # Load data
    print("Load data from", input_file)
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    print("Data loaded!")

    # Add PI to dict
    for i in tqdm(range(len(data))):
        d = data[i]
        PI_labels = generate_PI_labels(d['boxes'], d['center_bxs'], d['depth_med'], d['depth_mea'], d['depth_std'], d['depth_q25'], d['depth_q75']).astype(np.float32)
        #print(PI_labels.dtype)
        #break
        data[i]["PI_labels"] = PI_labels
        
    # Save data as pkl
    output_file = input_file.replace(replace_str[0], replace_str[1])
    with open(output_file, 'wb') as handle:
        pickle.dump(data, handle, protocol=4)
    print("Save data to", output_file)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None)
    args = parser.parse_args()

    add_pi_labels_to_pkl(args.input_file, replace_str = ("v2", "v3"))

