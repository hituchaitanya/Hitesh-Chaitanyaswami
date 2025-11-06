import numpy as np

def dice_coef(pred, true, eps=1e-7):
    pred = pred.flatten()
    true = true.flatten()
    inter = (pred * true).sum()
    return (2. * inter + eps) / (pred.sum() + true.sum() + eps)
