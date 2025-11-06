from src.segmentation import UNet
from src.recognition import CRNN

def get_segmentation(cfg):
    return UNet(in_channels=cfg['in_channels'], out_channels=cfg['out_channels'], base=cfg['base_channels'])

def get_recognition(cfg):
    return CRNN(imgH=cfg['img_h'], nc=cfg['num_channels'], nclass=cfg['num_classes'], nh=cfg['hidden_size'])
