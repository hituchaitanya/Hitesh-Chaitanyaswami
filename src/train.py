import yaml, os, random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import ChequeDataset
from src.model_zoo import get_segmentation
from tqdm import tqdm
import torch.nn as nn

def seed_everything(seed=42):
    import numpy as np
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train_segmentation(cfg):
    seed_everything(cfg['seed'])
    device = torch.device(cfg['device'])
    ds = ChequeDataset(cfg['data']['train_dir'], img_size=tuple(cfg['data']['img_size']))
    dl = DataLoader(ds, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    model = get_segmentation(cfg['seg']).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    criterion = nn.BCELoss()
    epochs = cfg['training']['epochs']
    ckpt_dir = cfg['training']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    for ep in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        for imgs, masks, _ in tqdm(dl):
            imgs = imgs.to(device)
            masks = masks.to(device)
            pred = model(imgs)
            loss = criterion(pred, masks)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        print(f"Epoch {ep}/{epochs} loss={epoch_loss/len(dl):.4f}")
        if ep % cfg['training']['save_every'] == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"seg_ep{ep}.pth"))

if __name__ == "__main__":
    cfg = yaml.safe_load(open('config.yaml'))
    train_segmentation(cfg)
