# src/preprocessing.py
import cv2
import numpy as np

def st_clahe(gray, clip=2.0, tile=(8,8)):
    """ST-CLAHE: CLAHE with a saturation-threshold inspired limiter.
       This keeps extreme intensity changes limited (simple implementation)."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    out = clahe.apply(gray)
    # simple squeeze step: blend original + clahe depending on local contrast
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    # compute local contrast map (absolute Laplacian)
    contrast = cv2.normalize(np.abs(lap), None, 0, 1.0, cv2.NORM_MINMAX)
    # invert contrast to get areas to preserve (faded areas -> higher blend)
    blend = (1.0 - contrast).astype(np.float32)
    blend = cv2.blur(blend, (15,15))
    out_f = (out.astype(np.float32) * blend + gray.astype(np.float32) * (1-blend)).astype(np.uint8)
    return out_f

def median_denoise(gray, k=3):
    return cv2.medianBlur(gray, k)

def normalize_img(img):
    arr = img.astype('float32')/255.0
    return arr
