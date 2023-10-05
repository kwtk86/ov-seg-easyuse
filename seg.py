# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# forked & modified by KWTK 202309 for easy use
# using CLIP

import multiprocessing as mp

import numpy as np
from PIL import Image
import os
import time



from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from open_vocab_seg import add_ovseg_config
from open_vocab_seg.utils import VisualizationDemo
import glob


# ckpt_url = 'https://drive.google.com/uc?id=1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy'
# output = './ovseg_swinbase_vitL14_ft_mpt.pth'
# gdown.download(ckpt_url, output, quiet=False)

def setup_cfg(config_file):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

class SegDef:
    def __init__(self, classes: str) -> None:
        pass

    def inference(self, img) -> np.ndarray:
        pass

mp.set_start_method("spawn", force=True)
config_file = './ovseg_swinB_vitL_demo.yaml'
cfg = setup_cfg(config_file)
demo = VisualizationDemo(cfg)



def inference(img, classes):
    class_names = classes.split(',')
    seg_mask = demo.run_on_image(img, class_names)
    return seg_mask

def save_mask(mask, out_path):
    out_img = np.empty((mask.shape[0], mask.shape[1], 3))
    out_img[mask==255] = [0,0,0]
    out_img[mask==0] = [128,0,0] # 这里可以做映射表，但20230826没做
    out_img[mask==1] = [128,0,0] # 这里可以做映射表，但20230826没做
    out_img[mask==2] = [0,0,128] # 这里可以做映射表，但20230826没做
    out_img[mask==3] = [0,128,0] # 这里可以做映射表，但20230826没做
    out_img = Image.fromarray(np.uint8(out_img)).convert('RGB')
    out_img.save(out_path)

def mask_oriimg(ori, mask, out_path):
    ori2 = ori.copy().astype(np.float16)
    ori2[mask==255]*=0.1
    ori2[mask==2]*=0.25
    ori2[mask==3]*=0.4
    ori2[mask==4]*=0.4
    out_img = Image.fromarray(np.uint8(ori2)).convert('RGB')
    out_img.save(out_path)


for pic in glob.glob('./pics4-p/*/*.png'):
    st = time.time()
    # building
    out_path = pic.replace('./pics4-p', './picsoutbs-p')
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img = read_image(pic, format="BGR")
    mask = inference(img, 'building,house,sky,plants')
    # mask_oriimg(img[:,:,::-1], mask, out_path)
    save_mask(mask, out_path)
    # sky
    # out_path = pic.replace('./pics4', './picsouts')
    # out_dir = os.path.dirname(out_path)
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    # img = read_image(pic, format="BGR")
    # mask = inference(img, 'sky')
    # save_mask(mask, out_path)
    # # plants
    # out_path = pic.replace('./pics4', './picsoutp')
    # out_dir = os.path.dirname(out_path)
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    # img = read_image(pic, format="BGR")
    # mask = inference(img, 'plants')
    # save_mask(mask, out_path)

    print(pic, time.time() - st)
