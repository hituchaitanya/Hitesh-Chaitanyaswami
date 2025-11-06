import cv2
import numpy as np

def read_gray(path, target_size=None):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if target_size is not None:
        im = cv2.resize(im, (target_size[1], target_size[0]))
    return im

def enhance_contrast(img, clip_lim=2.0, tile=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=tile)
    return clahe.apply(img)

def denoise(img, h=10):
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)

def normalize(img):
    img = img.astype('float32') / 255.0
    return img
