import yaml, os, cv2, torch
from src.preprocessing import read_gray, enhance_contrast, denoise, normalize
from src.model_zoo import get_segmentation, get_recognition
from src.refinement import canny_refine
import numpy as np

cfg = yaml.safe_load(open('config.yaml'))
device = torch.device(cfg['device'])

seg = get_segmentation(cfg['seg']).to(device)
seg.load_state_dict(torch.load(os.path.join(cfg['training']['checkpoint_dir'],'seg_ep5.pth'), map_location=device))
seg.eval()

def pipeline(image_path):
    im = read_gray(image_path, target_size=tuple(cfg['data']['img_size']))
    im = enhance_contrast(im, clip_lim=cfg['preproc']['contrast_clip_limit'], tile=tuple(cfg['preproc']['clahe_tile']))
    im = denoise(im, h=cfg['preproc']['denoise_h'])
    inp = normalize(im)
    inp_t = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        mask = seg(inp_t).cpu().numpy()[0,0]
    refined = canny_refine(mask)
    return im, mask, refined
