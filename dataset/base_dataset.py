from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
import torch

class BaseDataset(Dataset):
    def __init__(self, opt):
        self.crop_size = 512
        self.debug_mode = opt.debug_mode
        self.data_path = opt.data_path # dataset path. e.g., ./data/
        self.camera_name = opt.camera 
        self.gamma = opt.gamma
        self.camera_params_file = getattr(opt, "camera_params", None)  # PATCHED (OpenRAW): see camera_params()

    def norm_img(self, img, max_value):
        img = img / float(max_value)        
        return img

    def pack_raw(self, raw):
        # pack Bayer image to 4 channels
        im = np.expand_dims(raw, axis=2)
        H, W = raw.shape[0], raw.shape[1]
        # RGBG
        out = np.concatenate((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)
        return out
    
    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)
    
    def center_crop(self, img, crop_size=None):
        H = img.shape[0]
        W = img.shape[1]

        if crop_size is not None:
            th, tw = crop_size[0], crop_size[1]
        else:
            th, tw = self.crop_size, self.crop_size
        x1_img = int(round((W - tw) / 2.))
        y1_img = int(round((H - th) / 2.))
        if img.ndim == 3:
            input_patch = img[y1_img:y1_img + th, x1_img:x1_img + tw, :]
        else:
            input_patch = img[y1_img:y1_img + th, x1_img:x1_img + tw]

        return input_patch

    def load(self, is_train=True):
        # ./data
        # ./data/NIKON D700/RAW, ./data/NIKON D700/RGB 
        # ./data/Canon EOS 5D/RAW,  ./data/Canon EOS 5D/RGB 
        # ./data/NIKON D700_train.txt, ./data/NIKON D700_test.txt 
        # ./data/NIKON D700_train.txt: a0016, ... 
        input_RAWs_WBs = [] 
        target_RGBs = []        
        self.camera_index = []  # PATCHED (OpenRAW): which --camera each pair came from
        
        data_path = self.data_path # ./data/
        # PATCHED (OpenRAW, not upstream): accepts several cameras (a list,
        # or a comma-separated string) and pools them into one dataset.
        # Upstream took exactly one camera name.
        cams = self.camera_name if isinstance(self.camera_name, (list, tuple)) else str(self.camera_name).split(",")
        self.camera_names = [c.strip() for c in cams if c.strip()]
        for cam_i, cam in enumerate(self.camera_names):
            txt_path = data_path + cam + ("_train.txt" if is_train else "_test.txt")
            with open(txt_path, "r") as f_read:
                valid_camera_list = [line.strip() for line in f_read.readlines() if line.strip()]
            if self.debug_mode:
                valid_camera_list = valid_camera_list[:10]
            skipped = 0
            for name in valid_camera_list:
                full_name = data_path + cam
                raw_p, rgb_p = full_name + "/RAW/" + name + ".npz", full_name + "/RGB/" + name + ".jpg"
                # skip listed images that were never prepared (partial downloads
                # with a published split list) instead of crashing mid-epoch
                if not (os.path.exists(raw_p) and os.path.exists(rgb_p)):
                    skipped += 1
                    continue
                input_RAWs_WBs.append(raw_p)
                target_RGBs.append(rgb_p)
                self.camera_index.append(cam_i)
            if skipped:
                print(f"[data] {cam}: skipped {skipped} listed image(s) that aren't prepared")

        return input_RAWs_WBs, target_RGBs 

    # PATCHED (OpenRAW, not upstream): camera settings (exposure time, f-number,
    # ISO, focal length) for EXIF conditioning. From the pair's .npz when it was
    # prepared with them, else from camera_params.json (data/camera_params.py),
    # else unknown (NaN).
    def camera_params(self, npz, raw_path):
        if "camera_params" in npz.files:
            return np.asarray(npz["camera_params"], np.float32)
        table = getattr(self, "_param_table", None)
        if table is None:
            import json
            table = {}
            for f in [getattr(self, "camera_params_file", None), os.path.join(self.data_path, "camera_params.json")]:
                if f and os.path.isfile(f):
                    with open(f) as fh:
                        table.update(json.load(fh))
            self._param_table = table
        parts = raw_path.replace("\\", "/").split("/")
        key = parts[-3] + "/" + os.path.splitext(parts[-1])[0] if len(parts) >= 3 else ""
        v = table.get(key)
        return np.asarray([np.nan if x is None else x for x in (v or [None] * 4)], np.float32)


    def __len__(self):
        return 0

    def __getitem__(self, idx):

        return None
