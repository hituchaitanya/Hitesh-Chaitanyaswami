# src/train.py - trains the classifier on provided images (segmentation/recognition steps can be inserted)
import yaml, os, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import ChequeLineDataset
from src.models import SGCIP_CNN
from tqdm import tqdm
import numpy as np

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train_classifier(cfg_path='config.yaml', data_dir='data/synth'):
    cfg = yaml.safe_load(open(cfg_path))
    seed_everything(cfg['seed'])
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    ds = ChequeLineDataset(data_dir, img_size=(32,32))
    dl = DataLoader(ds, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    # NOTE: for demo we create fake labels (replace with real labels)
    model = SGCIP_CNN(n_classes=cfg['clf']['n_classes']).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg['clf']['lr'])
    criterion = nn.CrossEntropyLoss()
    epochs = cfg['clf']['epochs']
    os.makedirs(cfg['clf']['checkpoint_dir'], exist_ok=True)
    for ep in range(1, epochs+1):
        model.train()
        losses=[]
        for imgs, names in tqdm(dl):
            imgs = imgs.to(device).float()
            # create dummy labels for synthetic dataset: use name parity to have class 0/1
            labels = torch.tensor([0 if int(n.split('.')[0])%2==0 else 1 for n in names], dtype=torch.long).to(device)
            # resize to 32x32 and channel
            imgs = torch.nn.functional.interpolate(imgs, size=(32,32))
            out = model(imgs)
            loss = criterion(out, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"Epoch {ep}/{epochs} loss={np.mean(losses):.4f}")
        if ep % cfg['clf']['save_every'] == 0:
            torch.save(model.state_dict(), os.path.join(cfg['clf']['checkpoint_dir'], f"sgcip_ep{ep}.pth"))

if __name__ == "__main__":
    train_classifier()
