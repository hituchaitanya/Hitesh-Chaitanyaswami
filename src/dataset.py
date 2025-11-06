# src/dataset.py
import os, cv2
import torch
from torch.utils.data import Dataset

class ChequeLineDataset(Dataset):
    def __init__(self, root_dir, img_size=(512,256)):
        self.root = root_dir
        self.files = sorted([os.path.join(root_dir,f) for f in os.listdir(root_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.size[1], self.size[0]))
        tensor = torch.from_numpy(img.astype('float32')/255.0).unsqueeze(0)
        name = os.path.basename(p)
        return tensor, name
