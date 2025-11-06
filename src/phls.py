# src/phls.py
import cv2
import numpy as np

def extract_pseudolettes(binary_line):
    """
    Heuristic pseudo-letter extraction:
    - find contours of connected components
    - for each component compute bounding box height
    - return list of (x,y,w,h,mask)
    """
    contours, _ = cv2.findContours(binary_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 20: 
            continue
        mask = np.zeros_like(binary_line)
        cv2.drawContours(mask, [c], -1, 255, -1)
        boxes.append((x,y,w,h,mask))
    # sort left-to-right
    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes

def phls_segment(line_img_gray, bin_thresh=180):
    """Given a line-level grayscale image, segment pseudo-letters and choose those
    that look handwriting-like by height variance heuristics."""
    _, bw = cv2.threshold(line_img_gray, bin_thresh, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    boxes = extract_pseudolettes(bw)
    heights = [b[3] for b in boxes]
    if len(heights)==0:
        return np.zeros_like(bw)
    avg_h = np.mean(heights)
    var_h = np.var(heights)
    results = np.zeros_like(bw)
    for (x,y,w,h,mask) in boxes:
        # Handwritten tends to have varied heights: accept if height not equal to modal printed height
        if h < avg_h*1.6 and h > avg_h*0.4:
            results = cv2.bitwise_or(results, mask)
        else:
            # keep possibility of small noisy components but not printed uniform text (we use variance threshold)
            if var_h > 0.5*avg_h: 
                results = cv2.bitwise_or(results, mask)
    # morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    results = cv2.morphologyEx(results, cv2.MORPH_CLOSE, kernel, iterations=1)
    return results
