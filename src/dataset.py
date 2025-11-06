import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
from src.preprocessing import read_gray, enhance_contrast, denoise, normalize

class ChequeDataset(Dataset):
    def __init__(self, root_dir, img_size=(512,512), transforms=None, mode='train'):
        self.root = root_dir
        self.imgs = sorted([p for p in os.listdir(root_dir) if p.endswith('.png') or p.endswith('.jpg')])
        self.img_size = img_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        name = self.imgs[idx]
        path = os.path.join(self.root, name)
        img = read_gray(path, target_size=self.img_size)
        img = enhance_contrast(img)
        img = denoise(img)
        img_n = normalize(img)
        tensor = torch.from_numpy(img_n).unsqueeze(0).float()
        # For segmentation ground truth, expect mask at same name with _mask
        mask_path = path.replace('.png','_mask.png').replace('.jpg','_mask.png')
        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (self.img_size[1], self.img_size[0]))
            m = (m>127).astype('float32')
            mask_t = torch.from_numpy(m).unsqueeze(0)
        else:
            mask_t = torch.zeros(1, *self.img_size, dtype=torch.float32)
        return tensor, mask_t, name
