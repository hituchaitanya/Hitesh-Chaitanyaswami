# src/inference.py
import yaml, os, cv2, torch
from src.preprocessing import st_clahe, median_denoise, normalize_img
from src.ssmhed import ssmhed
from src.phls import phls_segment
from src.inpainting import patch_based_inpaint
from src.models import SGCIP_CNN

cfg = yaml.safe_load(open('config.yaml'))
device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

def run_pipeline(image_path, model_ckpt=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (cfg['data']['img_size'][1], cfg['data']['img_size'][0]))
    pre = st_clahe(img, clip=cfg['preproc']['clahe_clip'], tile=tuple(cfg['preproc']['clahe_tile']))
    pre = median_denoise(pre, k=cfg['preproc']['median_ksize'])
    # segmentation using phls on the whole image lines (for demo we run on full image)
    ph = phls_segment(pre)
    # detect edges for faded strokes
    edges = ssmhed(pre, k=cfg['edge']['sobel_k'], lam=cfg['edge']['lambda_r'], threshold=cfg['edge']['threshold'])
    # build a fade mask: region where original is light and edges absent
    fade_mask = ((pre > 200) & (edges==0)).astype('uint8')*255
    restored = patch_based_inpaint(pre, fade_mask, patch_size=cfg['inpaint']['patch_size'])
    # prepare classifier input: crop a candidate area (this is demo)
    small = cv2.resize(restored, (32,32)).astype('float32')/255.0
    tensor = torch.from_numpy(small).unsqueeze(0).unsqueeze(0).to(device)
    model = SGCIP_CNN(n_classes=cfg['clf']['n_classes']).to(device)
    if model_ckpt:
        model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(dim=1).item()
    return {
        'preprocessed': pre,
        'phls_mask': ph,
        'edges': edges,
        'restored': restored,
        'pred_class': int(pred)
    }
