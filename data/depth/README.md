# Generate data for LXMERDT

We need LXMERT data (e.g. img_id, 36 objects with bounding box and 2000 dim. features) and original source images where we run MiDas on.
For each image in LXMERT data files we run a depth estimation on the source images and calculate the depth for each of the 36 objects (based on the bounding box sub-image).

We calculate several values for each object

* depth of middle value
* minimum depth
* maximum depth
* mean depth
* median depth 


LXMERT saves image data in `*.tsv` files, i.e.

* `data/mscoco_imgfeat/train2014_obj36.tsv` based on COCO's `train2014.zip`
* `data/mscoco_imgfeat/val2014_obj36.tsv`   based on COCO's `val2014.zip`  
* `data/mscoco_imgfeat/test2015_obj36.tsv`  based on COCO's `test2015.zip`
* `data/vg_gqa_imgfeat/vg_gqa_obj36.tsv`    based on GQA's  `gqa/images.zip`


## Download source images from COCO and VG

```bash
mkdir -p data/depth
# Download
wget http://images.cocodataset.org/zips/train2014.zip -P data/depth # 13 GB, 82784 images
wget http://images.cocodataset.org/zips/val2014.zip -P data/depth   # 6.3GB, 40505 images
wget http://images.cocodataset.org/zips/test2015.zip -P data/depth  # 13 GB, 81435 images
wget https://nlp.stanford.edu/data/gqa/images.zip -P data/depth     #

# Unzip and remove zip
unzip data/depth/train2014.zip -d data && rm data/depth/train2014.zip
unzip data/depth/val2014.zip -d data && rm data/depth/val2014.zip
unzip data/depth/test2015.zip -d data && rm data/depth/test2015.zip
```


It should look like this 

```bash
tree data/depth/
.
├── add_depth_information.py
├── train2014_obj36.pkl
├── val2014_obj36.pkl
├── vg_gqa_obj36.pkl
├── coco
│   ├── ...
│   ├── ...
│   └── ...
└── gqa
    ├── ...
    ├── ...
    └── ...
```


## Generate depth values 

For testing use `--nb_of_files 100`:

```bash
# train2014_obj36.tsv
python add_depth_information.py --obj_path_dir /root/lxmerdt/data/mscoco_imgfeat --obj_file train2014_obj36.tsv --img_path_dir /root/lxmerdt/data/depth/coco/train2014 --nb_of_files 100 --save_abs_filename /root/lxmerdt/data/depth/train2014_obj36.pkl
```

```bash
# val2014_obj36.tsv
python add_depth_information.py --obj_path_dir /root/lxmerdt/data/mscoco_imgfeat --obj_file val2014_obj36.tsv --img_path_dir /root/lxmerdt/data/depth/coco/val2014 --nb_of_files 100 --save_abs_filename /root/lxmerdt/data/depth/val2014_obj36.pkl
```

```bash
# vg_gqa_obj36.tsv
python add_depth_information.py --obj_path_dir /root/lxmerdt/data/vg_gqa_imgfeat --obj_file vg_gqa_obj36.tsv --img_path_dir /root/lxmerdt/data/depth/vggqa --alternative_prefix COCO_test2015_ --nb_of_files 100 --save_abs_filename /root/lxmerdt/data/depth/vg_gqa_obj36.pkl
```

* Full data sets:

```bash
# train2014_obj36.tsv
CUDA_VISIBLE_DEVICES=4 python add_depth_information.py --obj_path_dir /root/lxmerdt/data/mscoco_imgfeat --obj_file train2014_obj36.tsv --img_path_dir /root/lxmerdt/data/depth/coco/train2014 --nb_of_files -1 --save_abs_filename /root/lxmerdt/data/depth/train2014_obj36.pkl 2>&1 | tee train2014_obj36.log
```

```bash
# val2014_obj36.tsv
CUDA_VISIBLE_DEVICES=5 python add_depth_information.py --obj_path_dir /root/lxmerdt/data/mscoco_imgfeat --obj_file val2014_obj36.tsv --img_path_dir /root/lxmerdt/data/depth/coco/val2014 --nb_of_files -1 --save_abs_filename /root/lxmerdt/data/depth/val2014_obj36.pkl 2>&1 | tee val2014_obj36.log
```

```bash
# vg_gqa_obj36.tsv
CUDA_VISIBLE_DEVICES=6 python add_depth_information.py --obj_path_dir /root/lxmerdt/data/vg_gqa_imgfeat --obj_file vg_gqa_obj36.tsv --img_path_dir /root/lxmerdt/data/depth/vggqa --alternative_prefix COCO_test2015_ --nb_of_files -1 --save_abs_filename /root/lxmerdt/data/depth/vg_gqa_obj36.pkl 2>&1 | tee vg_gqa_obj36.log
```

```bash
CUDA_VISIBLE_DEVICES=3 python add_depth_information.py --obj_path_dir /root/lxmerdt/data/vg_gqa_imgfeat --obj_file gqa_testdev_obj36.tsv --img_path_dir /root/lxmerdt/data/depth/vggqa --alternative_prefix COCO_test2015_ --nb_of_files -1 --save_abs_filename /root/lxmerdt/data/depth/gqa_testdev_obj36_depth_depth_v2.pkl 2>&1 | tee gqa_testdev_obj36_depth_depth_v2.log
```


# Add MPE labels

Please use an other Docker container (since MiDaS relies on a newer PyTorch version).

```
docker run -it --gpus all --name lxmerdt_midas -e "TERM=xterm-256color" -v /lxmerdt_research/lxmerdt:/root/lxmerdt/ nvcr.io/nvidia/pytorch:21.06-py3 bash
```

```bash
time python -u add_mpe_labels.py --input_file mscoco_imgfeat/train2014_obj36_depth_v2.pkl   
time python -u add_mpe_labels.py --input_file mscoco_imgfeat/val2014_obj36_depth_v2.pkl     
time python -u add_mpe_labels.py --input_file mscoco_imgfeat/test2015_obj36_depth_v2.pkl    
time python -u add_mpe_labels.py --input_file mscoco_imgfeat/gqa_testdev_obj36.pkl    
```


