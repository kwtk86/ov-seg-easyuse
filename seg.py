# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# forked & modified by KWTK 202309 for easy use
# using CLIP

import multiprocessing as mp

import numpy as np
from PIL import Image
import os
from typing import Tuple

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from open_vocab_seg import add_ovseg_config
from open_vocab_seg.utils import VisualizationDemo


# ckpt_url = 'https://drive.google.com/uc?id=1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy'
# output = './ovseg_swinbase_vitL14_ft_mpt.pth'
# gdown.download(ckpt_url, output, quiet=False)


__all__ = ['OvSegEasyuse']

def setup_cfg(config_file):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

class OvSegEasyuse:
    def __init__(self,
                 class_definition: dict,
                 cfg_file: str = './ovseg_swinB_vitL_demo.yaml', 
                 ) -> None:
        assert len(class_definition) < 255, '一次性最多输入254个类别'
        self.class_names, self.class_colors = [], []
        for k, v in class_definition.items():
            self.class_names.append(k.strip())
            assert isinstance(v, list) and len(v) == 3, '颜色必须是长度为3的列表'
            self.class_colors.append(np.array(v))
        mp.set_start_method("spawn", force=True)
        cfg = setup_cfg(cfg_file)
        self.__demo = VisualizationDemo(cfg)


    def inference_and_save(self, 
                           img_path: str, 
                           out_path: str,
                           masked_input_path: str = None) -> np.ndarray:
        mask, img = self.inference(img_path, return_img = True)
        self.save_mask(mask, out_path)
        if masked_input_path:
            self.save_masked_input(mask, img, masked_input_path)
        return mask

    def save_masked_input(self, 
                          mask: np.ndarray, 
                          img: np.ndarray, 
                          out_path: str) -> None:
        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        img = img[:,:,::-1].copy()
        for i, (class_name, class_color) in enumerate(zip(self.class_names, self.class_colors)):
            indices = (mask == i)
            img[indices] = img[indices]*0.5 + class_color*0.5 
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        img.save(out_path)


    def inference(self, 
                  img_path: str,
                  return_img: bool = False) -> np.ndarray or Tuple[np.ndarray, np.ndarray]:
        assert os.path.exists(img_path), f"{img_path} 不存在"
        img = read_image(img_path, format="BGR")
        seg_mask = self.__demo.run_on_image(img, self.class_names)
        if return_img:
            return seg_mask, img
        else:
            return seg_mask
    
    def save_mask(self, 
                  mask: np.ndarray, 
                  out_path: str) -> None:
        out_img = np.empty((mask.shape[0], mask.shape[1], 3))
        out_img[mask==255] = [0,0,0]
        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for i, (class_name, class_color) in enumerate(zip(self.class_names, self.class_colors)):
            out_img[mask == i] = class_color
        out_img = Image.fromarray(np.uint8(out_img)).convert('RGB')
        out_img.save(out_path)
