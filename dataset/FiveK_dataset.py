from __future__ import print_function, division
import os, random, time
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
import rawpy
from glob import glob
from PIL import Image as PILImage
import numbers
from .base_dataset import BaseDataset
from . import mosaic_store


# PATCHED (OpenRAW, not upstream): the original here was
#   from scipy.misc import imread
# scipy.misc.imread was removed from scipy years ago (deprecated ~2018,
# gone entirely by the scipy version this project actually has
# installed -- `from scipy.misc import imread` raises ImportError
# outright, not a deprecation warning). This is a drop-in behavioral
# equivalent for the plain RGB-JPEG case this file actually uses it for
# (data_preprocess.py's own output, always a standard JPEG) using PIL,
# which this file already imports.
def imread(path):
    return np.array(PILImage.open(path).convert("RGB"))


# PATCHED (OpenRAW, not upstream): normalize by each image's real white level.
# Upstream hardcoded 4095 for the Canon EOS 5D and 16383 for every other
# camera -- wrong for any other 12-bit sensor. data/fivek_prepare.py stores
# 'white_level' (black-subtracted) per image; files made by the legacy
# data/data_preprocess.py lack it and keep upstream's exact behavior.
def _norm_value(npz, raw_path, gamma):
    if "white_level" in npz.files:
        v = float(npz["white_level"])
    else:
        v = 4095.0 if "/Canon_EOS_5D/" in raw_path.replace("\\", "/") else 16383.0
    return np.power(v, 1 / 2.2) if gamma else v


class FiveKDatasetTrain(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt=opt) 
        self.patch_size = 256
        input_RAWs_WBs, target_RGBs = self.load(is_train=True)
        assert len(input_RAWs_WBs) == len(target_RGBs)        
        self.data = {'input_RAWs_WBs':input_RAWs_WBs, 'target_RGBs':target_RGBs} 

    def random_flip(self, input_raw, target_rgb):
        idx = np.random.randint(2)
        input_raw = np.flip(input_raw,axis=idx).copy()
        target_rgb = np.flip(target_rgb,axis=idx).copy()
        
        return input_raw, target_rgb

    def random_rotate(self, input_raw, target_rgb):
        idx = np.random.randint(4)
        input_raw = np.rot90(input_raw,k=idx)
        target_rgb = np.rot90(target_rgb,k=idx)

        return input_raw, target_rgb

    def random_crop(self, patch_size, input_raw, target_rgb,flow=False,demos=False):
        H, W, _ = input_raw.shape
        rnd_h = random.randint(0, max(0, H - patch_size))
        rnd_w = random.randint(0, max(0, W - patch_size))

        patch_input_raw = input_raw[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
        if flow or demos:
            patch_target_rgb = target_rgb[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
        else:
            patch_target_rgb = target_rgb[rnd_h*2:rnd_h*2 + patch_size*2, rnd_w*2:rnd_w*2 + patch_size*2, :]

        return patch_input_raw, patch_target_rgb
        
    def aug(self, patch_size, input_raw, target_rgb, flow=False, demos=False):
        input_raw, target_rgb = self.random_crop(patch_size, input_raw,target_rgb,flow=flow, demos=demos)
        input_raw, target_rgb = self.random_rotate(input_raw,target_rgb)
        input_raw, target_rgb = self.random_flip(input_raw,target_rgb)
        
        return input_raw, target_rgb

    def __len__(self):
        return len(self.data['input_RAWs_WBs'])

    def __getitem__(self, idx):    
        input_raw_wb_path = self.data['input_RAWs_WBs'][idx]
        target_rgb_path = self.data['target_RGBs'][idx]
        
        target_rgb_img = imread(target_rgb_path)
        input_raw_wb = np.load(input_raw_wb_path)
        wb = input_raw_wb['wb']
        wb = wb / wb.max() 
        self.patch_size = 256

        if mosaic_store.is_mosaic(input_raw_wb):
            # PATCHED (OpenRAW, not upstream): compact mosaic storage (see
            # dataset/mosaic_store.py). Pick the crop first, then demosaic only
            # that region -- ~1% of a full-frame demosaic for a 256px patch.
            # Crop origin is even so the CFA phase is preserved; result is
            # identical to demosaicing the full frame and cropping.
            mosaic, pattern = mosaic_store.unpack(input_raw_wb)
            H = min(mosaic.shape[0], target_rgb_img.shape[0])
            W = min(mosaic.shape[1], target_rgb_img.shape[1])
            ps = self.patch_size
            y = random.randint(0, max(0, H - ps)) // 2 * 2
            x = random.randint(0, max(0, W - ps)) // 2 * 2
            h, w = min(ps, H - y), min(ps, W - x)
            input_raw_img = mosaic_store.demosaic_region(mosaic, pattern, y, x, h, w)
            np.clip(input_raw_img, 0, float(input_raw_wb['white_level']), out=input_raw_img)
            target_rgb_img = target_rgb_img[y:y + h, x:x + w]
            input_raw_img = input_raw_img * wb[:-1]
            input_raw_img, target_rgb_img = self.random_rotate(input_raw_img, target_rgb_img)
            input_raw_img, target_rgb_img = self.random_flip(input_raw_img, target_rgb_img)
        else:
            input_raw_img = input_raw_wb['raw']
            input_raw_img = input_raw_img * wb[:-1]   
            input_raw_img, target_rgb_img = self.aug(self.patch_size, input_raw_img, target_rgb_img, flow=True, demos=True)  

        norm_value = _norm_value(input_raw_wb, input_raw_wb_path, self.gamma)  # PATCHED, see _norm_value
        if self.gamma:
            input_raw_img = np.power(input_raw_img, 1/2.2)

        target_rgb_img = self.norm_img(target_rgb_img, max_value=255)
        input_raw_img = self.norm_img(input_raw_img, max_value=norm_value)   
        target_raw_img = input_raw_img.copy()

        input_raw_img = self.np2tensor(input_raw_img).float()
        target_rgb_img = self.np2tensor(target_rgb_img).float()
        target_raw_img = self.np2tensor(target_raw_img).float()
        
        sample = {'input_raw':input_raw_img, 'target_rgb':target_rgb_img, 'target_raw':target_raw_img,
                    'file_name':input_raw_wb_path.split("/")[-1].split(".")[0]}
        return sample

class FiveKDatasetTest(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt=opt)
        self.patch_size = 256
        
        input_RAWs_WBs, target_RGBs = self.load(is_train=False)
        assert len(input_RAWs_WBs) == len(target_RGBs)        
        self.data = {'input_RAWs_WBs':input_RAWs_WBs, 'target_RGBs':target_RGBs} 

    def __len__(self):
        return len(self.data['input_RAWs_WBs'])

    def __getitem__(self, idx):    
        input_raw_wb_path = self.data['input_RAWs_WBs'][idx]
        target_rgb_path = self.data['target_RGBs'][idx]
        
        target_rgb_img = imread(target_rgb_path)
        input_raw_wb = np.load(input_raw_wb_path)
        # PATCHED (OpenRAW): optional deterministic centre crop (eval_crop, set by
        # train.py's evaluation): full frames don't fit a small GPU and are slow
        crop = getattr(self, "eval_crop", None)
        if mosaic_store.is_mosaic(input_raw_wb):  # PATCHED (OpenRAW): compact storage
            mosaic, pattern = mosaic_store.unpack(input_raw_wb)
            if crop:
                H = min(mosaic.shape[0], target_rgb_img.shape[0]); W = min(mosaic.shape[1], target_rgb_img.shape[1])
                h, w = min(crop, H) // 2 * 2, min(crop, W) // 2 * 2
                y, x = (H - h) // 4 * 2, (W - w) // 4 * 2  # centred, even (keeps the CFA phase)
                input_raw_img = mosaic_store.demosaic_region(mosaic, pattern, y, x, h, w)
                target_rgb_img = target_rgb_img[y:y + h, x:x + w]
            else:
                input_raw_img = mosaic_store.demosaic(mosaic, pattern)
            np.clip(input_raw_img, 0, float(input_raw_wb['white_level']), out=input_raw_img)
        else:
            input_raw_img = input_raw_wb['raw']
            if crop:
                H = min(input_raw_img.shape[0], target_rgb_img.shape[0]); W = min(input_raw_img.shape[1], target_rgb_img.shape[1])
                h, w = min(crop, H) // 2 * 2, min(crop, W) // 2 * 2
                y, x = (H - h) // 4 * 2, (W - w) // 4 * 2
                input_raw_img = input_raw_img[y:y + h, x:x + w]
                target_rgb_img = target_rgb_img[y:y + h, x:x + w]
        wb = input_raw_wb['wb']
        wb = wb / wb.max() 
        input_raw_img = input_raw_img * wb[:-1]   

        norm_value = _norm_value(input_raw_wb, input_raw_wb_path, self.gamma)  # PATCHED, see _norm_value
        if self.gamma:
            input_raw_img = np.power(input_raw_img, 1/2.2)

        target_rgb_img = self.norm_img(target_rgb_img, max_value=255)
        input_raw_img = self.norm_img(input_raw_img, max_value=norm_value)   
        target_raw_img = input_raw_img.copy()

        input_raw_img = self.np2tensor(input_raw_img).float()
        target_rgb_img = self.np2tensor(target_rgb_img).float()
        target_raw_img = self.np2tensor(target_raw_img).float()
        
        sample = {'input_raw':input_raw_img, 'target_rgb':target_rgb_img, 'target_raw':target_raw_img,
                    'file_name':input_raw_wb_path.split("/")[-1].split(".")[0]}
        return sample

