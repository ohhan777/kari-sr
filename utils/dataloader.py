import os
import glob
from osgeo import gdal
import hashlib
from pathlib import Path
import numpy as np
import random
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from utils.torch_utils import torch_distributed_zero_first
from utils.augmentations import Albumentations, letterbox, random_perspective, augment_hsv, random_scale, random_crop

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def create_dataloader(is_train, batch_size, hyp=None, augment=False,
                      cache=False, pad=0.0, rank=-1, workers=8,
                      shuffle=True):
    #with torch_distributed_zero_first(rank):
    dataset = LoadImages(is_train, batch_size, augment=augment,
                                      hyp=hyp, cache_imgs=cache)
    batch_size = min(batch_size, len(dataset))
    num_devices = torch.cuda.device_count()
    num_workers = min([os.cpu_count() // max(num_devices, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
                        num_workers=num_workers, sampler=sampler, pin_memory=True, drop_last=False)
    return loader, dataset


class LoadImages(Dataset):
    cache_version = 0.1   # dataset labels *.cache version

    def __init__(self, is_train=True, batch_size=16, augment=False, 
                 hyp=None, cache_imgs=False):

        self.augment = augment
        self.hyp = hyp
        self.is_train = is_train
        self.path = './data/wv3-k3/train' if is_train else './data/wv3-k3/train'
        #self.path = './data/k3a-k3/train' if is_train else './data/k3a-k3/train'
        self.scale_factor = 2
        self.crop_size = tuple(hyp['crop_size'])
        self.source_mean = hyp['lr_mean']
        self.source_std = hyp['lr_std']
        self.target_mean = hyp['hr_mean']
        self.target_std = hyp['hr_std']
        self.source_gsd = hyp['lr_gsd']
        self.hr_gsd = hyp['hr_gsd']
        self.lr_bits = hyp['lr_bits']
        self.hr_bits = hyp['hr_bits']


        self.hr_files = glob.glob(os.path.join(self.path, 'WV3/*.tif'))
        #self.hr_files = glob.glob(os.path.join(self.path, 'K3A/*.tif'))
        self.num_hr_files = len(self.hr_files)
        self.num_files = self.num_hr_files
        if is_train:
            self.lr_files = glob.glob(os.path.join(self.path, 'K3/*.tif'))
            #self.lr_files = glob.glob(os.path.join(self.path, 'K3/*.tif'))
            self.num_lr_files = len(self.lr_files)
            self.num_files = max(self.num_hr_files, self.num_lr_files)

    
    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        hr_file = self.hr_files[index % self.num_hr_files]
        
        #hr_img = self.open_geotiff(hr_file, "G")
        hr_img = self.open_geotiff(hr_file)
        dshr_img, hr_img = self.hr_transform(hr_img)
        
        if self.is_train:
            lr_file = self.lr_files[index % self.num_lr_files]
            #lr_img = self.open_geotiff(lr_file, "G")
            lr_img = self.open_geotiff(lr_file)
            lr_img = self.lr_transform(lr_img)
            return dshr_img.copy(), lr_img.copy(), hr_img.copy() , hr_file
        else:
            return lr_img.copy()


    def hr_transform(self, img):
        h, w = img.shape[:2]

        # upsampling
        new_h = np.int(h * self.scale_factor * self.hr_gsd / self.source_gsd)
        new_w = np.int(w * self.scale_factor * self.hr_gsd / self.source_gsd)

        img = cv2.resize(img, (new_w, new_h),
                         interpolation=cv2.INTER_CUBIC)
        # random cropping
        h = self.scale_factor * self.crop_size[0]
        w = self.scale_factor * self.crop_size[1]
        img = self.random_crop(img, (h, w))
        
        # downsampling 
        ds_img = cv2.resize(img, self.crop_size, interpolation=cv2.INTER_CUBIC)
               
        # normalization
        img = img.astype(np.float32)/(2**self.hr_bits - 1)
        ds_img = ds_img.astype(np.float32)/(2**self.hr_bits - 1)        
        #img = (img - self.target_mean)/self.target_std

        return np.expand_dims(ds_img, axis=0), np.expand_dims(img, axis=0)
        

    def lr_transform(self, img):
        # random crop
        h = self.crop_size[0]
        w = self.crop_size[1]
        img = self.random_crop(img, (h, w))
        # normalization
        img = img.astype(np.float32)/(2**self.lr_bits - 1)

        
        # img = (img.astype(np.float32) - self.lr_mean)/self.lr_std

        return np.expand_dims(img, axis=0)

    def open_geotiff(self, path, band=None):
        ds = gdal.Open(path)
        assert ds != None, f'{path} is not found'
        return ds.GetRasterBand(1).ReadAsArray().astype(np.uint16)
        

    def random_crop(self, img, crop_size):
        h, w = img.shape
        img = self.pad_image(img, h, w, crop_size,
                               (0.0,))
        new_h, new_w = img.shape
        x = random.randint(0, new_w - crop_size[1])
        y = random.randint(0, new_h - crop_size[0])
        img = img[y:y+crop_size[0], x:x+crop_size[1]]

        return img

    def pad_image(self, image, h, w, size, pad_value):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=pad_value)

        return pad_image

  
            
    # @staticmethod
    # def collate_fn(batch):
    #     img, label = zip(*batch)  # transposed
    #     return torch.stack(img, 0), torch.stack(label, 0)    


