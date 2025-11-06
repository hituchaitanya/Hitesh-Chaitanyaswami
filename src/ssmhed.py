# src/ssmhed.py
import cv2
import numpy as np

def sobel_gradients(img, ksize=3):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx*gx + gy*gy)
    ang = np.arctan2(gy, gx)
    return gx, gy, mag, ang

def sparse_refine(mag, lam=0.8):
    """Simple sparse refinement: soft-thresholding on gradients to keep thin strokes."""
    # normalize
    m = mag.copy()
    if m.max() > 0:
        m = m / (m.max())
    # soft threshold
    thr = np.mean(m) * lam
    m_ref = np.where(m > thr, m, m*0.2)
    # enhance peaks
    m_ref = cv2.GaussianBlur(m_ref, (3,3), 0)
    return m_ref

def marr_hildreth_like(mag):
    """Approximate Marr-Hildreth by laplacian of gaussian on gradient magnitude."""
    g = cv2.GaussianBlur(mag, (5,5), 1.0)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    lap = np.abs(lap)
    lap = lap / (lap.max() + 1e-9)
    return lap

def ssmhed(img_gray, k=3, lam=0.8, threshold=0.12):
    # expect uint8 input 0-255
    gx, gy, mag, ang = sobel_gradients(img_gray, k)
    mag_ref = sparse_refine(mag, lam)
    lap = marr_hildreth_like(mag_ref)
    comb = mag_ref + lap
    comb = comb / (comb.max()+1e-9)
    edges = (comb > threshold).astype('uint8')*255
    # thin the edges
    edges = cv2.ximgproc.thinning(edges) if hasattr(cv2, 'ximgproc') else cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
    return edges
