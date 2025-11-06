import cv2
import numpy as np

def canny_refine(mask, low=50, high=150):
    mask_u8 = (mask*255).astype('uint8')
    edges = cv2.Canny(mask_u8, low, high)
    kernel = np.ones((3,3), np.uint8)
    dil = cv2.dilate(edges, kernel, iterations=1)
    refined = cv2.bitwise_or(mask_u8, dil)
    refined = (refined>127).astype('uint8')*255
    return refined
