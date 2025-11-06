import numpy as np

def dice(pred, target, eps=1e-7):
    p = pred.flatten()
    t = target.flatten()
    inter = (p*t).sum()
    return (2*inter + eps) / (p.sum() + t.sum() + eps)
