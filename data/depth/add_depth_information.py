#!/usr/bin/env python
# coding: utf-8

# # Depth

import os
import requests
import cv2
import torch
import torchvision
import urllib.request
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import pickle

print("torch is ", torch.__version__, ' should be 1.7.0')
print("torchvision is ", torchvision.__version__, 'should be 0.8.1')
print("cv2 is ", cv2.__version__, 'should be 4.1.2')

#from utils import load_obj_tsv
import sys
sys.path.append('/root/plxmert/src/')
from utils import load_obj_tsv

# Functions
def get_absolute_filename(path_dir, filename, extension=".jpg"):
    return "{}{}".format(os.path.join(path_dir, filename), extension)

def load_rgb_img(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def make_midas_depth_estimation(midas, transform, img, plot=False, device="cpu"):
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    if plot:
        plt.imshow(output)
    return output

def utils_get_data(query, delim=","):
    assert isinstance(query, str)
    if os.path.isfile(query):
        with open(query) as f:
            data = eval(f.read())
    else:
        req = requests.get(query)
        try:
            data = requests.json()
        except Exception:
            data = req.content.decode()
            assert data is not None, "could not connect"
            try:
                data = eval(data)
            except Exception:
                data = data.split("\n")
        req.close()
    return data
    
OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
print(f"Start to download from {OBJ_URL}...", end =" ")
objids = utils_get_data(OBJ_URL)
print("Downloading done.")

def calc_center(img2d_shape, x1,y1,x2,y2):
    center_y = int(y1 + (y2-y1)/2)
    center_x = int(x1 + (x2-x1)/2)

    # make sure values are in range
    if center_y < 0: center_y = 0
    if center_y > img2d_shape[0]: center_y = img2d_shape[0]
    if center_x < 0: center_x = 0
    if center_x > img2d_shape[1]: center_x = img2d_shape[1]
        
    return center_y, center_x


def scale_depth_img(img_depth):
    """Scale values from 2D image to be between 0 (=nearest) and 1 (=farest)"""
    vmin = img_depth.min()
    img_depth = img_depth - vmin
    vmax = img_depth.max()
    img_depth = 1 - img_depth/vmax
    return img_depth

# Main
def main(model, transform, obj_path_dir, obj_file_name, img_path_dir, alternative_prefix, nb_of_files=100, device='cpu', save_abs_filename=None, verbose=False):
    """

    """


    # Load tsv
    tsv_filename = os.path.join(obj_path_dir, obj_file_name)
    if nb_of_files < 0:
        nb_of_files = None
    data = load_obj_tsv(tsv_filename, nb_of_files)
    obj_file_base = os.path.basename(obj_file_name) # e.g. train2014_obj36.tsv to train2014_obj36
    img_nb = len(data) 


    # Iterate over each row 
    for ii in tqdm(range(img_nb), desc="Run depth estimator"):
        d = data[ii]
        img_id = d['img_id']


        if alternative_prefix: # COCO_test2015_
	        if img_id[0] == "n":
	        	# COCO /coco/test2015/ images
	        	img_name = img_id[1:] # rm leading "n"
	        	img_name = "{}{:012d}".format(alternative_prefix, int(img_name))
	        	filename = get_absolute_filename(img_path_dir, img_name)
	        else:
	        	# vggqa images
	        	filename = get_absolute_filename(img_path_dir, img_id)
        else:
	        filename = get_absolute_filename(img_path_dir, img_id)
        if verbose:
            print(ii, filename, end=" ")

        # Make depth estimation
        try:
            img = load_rgb_img(filename)
            img_depth = make_midas_depth_estimation(model, transform, img, plot=False, device=device)
            img_depth = scale_depth_img(img_depth) # Scale depth img (0=nearest, 1=farest)
        except Exception as e:
            print("ERROR", filename, e)
            raise

        # plot
        if verbose:
            _img = load_rgb_img(filename)
            plt.imshow(_img)
            plt.show()
            plt.imshow(img_depth)
            plt.show()

        # Create blank arrays 
        num_boxes = d['num_boxes']
        center_bxs = np.zeros((num_boxes, 2))
        depth_mid = np.zeros(num_boxes)
        depth_min = np.zeros(num_boxes)
        depth_max = np.zeros(num_boxes)
        depth_mea = np.zeros(num_boxes)
        depth_med = np.zeros(num_boxes)
        
        depth_std = np.zeros(num_boxes)
        depth_q25 = np.zeros(num_boxes)
        depth_q75 = np.zeros(num_boxes)

        for obj_idx in range(num_boxes):
            box = d['boxes'][obj_idx]
            object_id = d['objects_id'][obj_idx]
            object_label = objids[object_id]
            hight, width = img_depth.shape

            # Extract coordinates
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            img_box_depth = img_depth[y1:y2,x1:x2]

            # Calc center value
            center_y, center_x = calc_center(img_depth.shape, x1, y1, x2, y2)
            center_bxs[obj_idx] = [center_x, center_y] #save in order xy

            # Calc depth values for middle point, min depth, max depth, mean depth and median depth
            depth_mid[obj_idx] = img_depth[center_y, center_x] # extract in order yx
            depth_min[obj_idx] = img_box_depth.min() 
            depth_max[obj_idx] = img_box_depth.max()
            depth_mea[obj_idx] = img_box_depth.mean()
            depth_med[obj_idx] = np.median(img_box_depth)

            depth_std[obj_idx] = np.std(img_box_depth)
            depth_q25[obj_idx] = np.quantile(img_box_depth, 0.25)
            depth_q75[obj_idx] = np.quantile(img_box_depth, 0.75)

            if verbose:
                print("object_label:", object_label)
                cv2.rectangle(_img, (x1,y1), (x2,y2), (255,255,0), 1)
                cv2.putText(_img, object_label, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                plt.imshow(_img)
                plt.show()
                print("cent:", depth_mid[obj_idx], "min:", depth_min[obj_idx], "max:", depth_max[obj_idx], "mean:", depth_mea[obj_idx], "median:", depth_med[obj_idx])
                plt.imshow(img_box_depth, vmin=0, vmax=1)
                plt.show()

        # Add to 'data' dict
        data[ii]['depth_mid'] = depth_mid
        data[ii]['depth_min'] = depth_min
        data[ii]['depth_max'] = depth_max
        data[ii]['depth_mea'] = depth_mea
        data[ii]['depth_med'] = depth_med
        data[ii]['center_bxs'] = center_bxs

        data[ii]['depth_std'] = depth_std
        data[ii]['depth_q25'] = depth_q25
        data[ii]['depth_q75'] = depth_q75

        if verbose:
            print("_"*50)

    # After each image is processed
    if save_abs_filename:
        # Save as pkl
        print(f"Start to save data to {save_abs_filename}.")
        with open(save_abs_filename, 'wb') as f:
            pickle.dump(data, f)

    return data


# ## Run

#obj_path_dir = "/home/myname/Data/plxmert/data/mscoco_imgfeat/"
#obj_file = "val2014_obj36.tsv"
#img_path_dir = "/home/myname/Data/coco_val2014/val2014"
#data_new = main(obj_path_dir, obj_file, img_path_dir)


# ## Main

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Add depth info.')
    parser.add_argument('--obj_path_dir', dest='obj_path_dir', help='path to object file (.tsv)', default='mscoco_imgfeat', type=str)
    parser.add_argument('--obj_file', dest='obj_file', help='file to object file (.tsv)', default="val2014_obj36.tsv", type=str)
    parser.add_argument('--img_path_dir', dest='img_path_dir', help='path to img files', default="coco/val2014", type=str)
    parser.add_argument('--alternative_prefix', dest='alternative_prefix', help='alternative_prefix for vg_gqa_imgfeat/', default=None, type=str)
    parser.add_argument('--nb_of_files', dest='nb_of_files', help='number of loaded files (mainly for testing)', default=10, type=int)
    parser.add_argument('--save_abs_filename', dest='save_abs_filename', help='save path', default=None, type=str)
    parser.add_argument('--verbose', dest='verbose', help='verbosity', default=False, type=bool)

    args = parser.parse_args()
    return args

    
if __name__ == '__main__':
    
    # Load midas model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform
    print("Model loaded.")

    # Parse argumgents    
    args = parse_args()
    print(args)

    # Run depth estimator
    print("Run depth estimator:")
    data_new = main(model, transform, device=device, # model
        obj_path_dir=args.obj_path_dir, obj_file_name=args.obj_file, # lxmert data
        img_path_dir=args.img_path_dir, nb_of_files=args.nb_of_files, # source data
        alternative_prefix=args.alternative_prefix,
        save_abs_filename=args.save_abs_filename, verbose=args.verbose)
    print("Done!")

