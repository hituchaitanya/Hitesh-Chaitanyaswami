import torch
import yaml
from src.dataset import ChequeDataset
from src.model_zoo import get_segmentation
from torch.utils.data import DataLoader
import numpy as np
from utils.metrics import dice_coef

def evaluate_seg(ckpt_path):
    cfg = yaml.safe_load(open('config.yaml'))
    device = torch.device(cfg['device'])
    model = get_segmentation(cfg['seg'])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    ds = ChequeDataset(cfg['data']['test_dir'], img_size=tuple(cfg['data']['img_size']))
    dl = DataLoader(ds, batch_size=1)
    dices = []
    for imgs, masks, _ in dl:
        with torch.no_grad():
            pred = model(imgs.to(device)).cpu().numpy()
        d = dice_coef((pred>0.5).astype(int), masks.numpy().astype(int))
        dices.append(d)
    print("Mean Dice:", np.mean(dices))
