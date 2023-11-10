import os
import os.path as osp
# from pathlib import Path
import sys
import random
import torch
import torchvision
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from utils import *
import torchvision.transforms as transforms


class CityscapesDataset(data.Dataset):
    def __init__(self, label_root, rgb_root, label_path, rgb_path, crop_size, ignore_label=19):
        '''
        label_root - root path to label images
        rgb_root - root path to RGB images
        label_path - path to list of labels (label_root and paths in this file will be concatenated to get the full path)
        rgb_path - path to list of RGB images (rgb_root and paths in this file will be concatenated to get the full path)
        crop_size - size of images and labels required
        '''
        self.ignore_label = ignore_label
        self.label_root = label_root
        self.rgb_root = rgb_root
        self.img_ids = [i_id.strip() for i_id in open(label_path)]
        self.img_rgb_ids = [i_id.strip() for i_id in open(rgb_path)]
        self.crop_size = crop_size
        self.IMG_MEAN = np.asarray((104.00699, 116.66877, 122.67892), np.float32)
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        rgb = Image.open(os.path.join(self.rgb_root, self.img_rgb_ids[index]))
        label = Image.open(os.path.join(self.label_root, self.img_ids[index]))

        label = label.resize(self.crop_size, Image.NEAREST)

        label = np.asarray(label, dtype=np.uint8)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)

        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        rgb = rgb.resize(self.crop_size, Image.BICUBIC)

        rgb = np.asarray(rgb, np.float32)
        rgb = rgb[:, :, ::-1]  # change to BGR
        rgb -= self.IMG_MEAN
        rgb = rgb.transpose((2, 0, 1)).copy() # (C x H x W)
        return label_copy.copy(), rgb.copy()


class CityscapesDatasetSSL(data.Dataset):
    def __init__(self, root, list_path, crop_size=(11, 11), label_folder=None, mirror=False):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.label_folder = label_folder
        self.mirror = mirror

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(osp.join(self.root, "leftImg8bit/%s" % (name))).convert('RGB')
        # label = Image.open(self.label_folder+"/%s" % name.split('/')[2])
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)

        image = np.asarray(image, np.float32)
        # label = np.asarray(label, np.float32)

        size = image.shape
        image = image.transpose((2, 0, 1))

        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            # label = label[:, ::flip]

        # return label.copy(), image.copy()
        return None, image.copy()
