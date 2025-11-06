# src/inpainting.py
import cv2
import numpy as np

def patch_based_inpaint(img_gray, mask, patch_size=9, search_radius=30):
    """
    Simple exemplar-based inpainting on mask areas using OpenCV's TELEA as fallback.
    This function tries OpenCV's inpaint; for stronger restoration, a full exemplar
    algorithm can be plugged in.
    """
    if mask.sum() == 0:
        return img_gray
    # OpenCV inpaint expects 8-bit 1-channel mask (non-zero == inpaint)
    mask8 = (mask>0).astype('uint8')*255
    try:
        res = cv2.inpaint(img_gray, mask8, 3, cv2.INPAINT_TELEA)
    except Exception:
        res = cv2.inpaint(img_gray, mask8, 3, cv2.INPAINT_NS)
    return res
