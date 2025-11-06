# generate simple synthetic faded/overlap lines for reproducibility
import os, cv2, numpy as np, random
from PIL import Image, ImageDraw, ImageFont

def random_text_image(text, w=512, h=128, font_size=36, fade_prob=0.3):
    im = Image.new('L', (w,h), color=255)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    x=5; y=random.randint(5, h-font_size-5)
    draw.text((x,y), text, font=font, fill=0)
    arr = np.array(im).astype(np.uint8)
    if random.random() < fade_prob:
        # apply fade gradient
        mask = np.linspace(1,0.2,w).astype(np.float32)
        arr = (arr * mask[np.newaxis,:]).astype(np.uint8)
    # add overlapping strokes from another line
    if random.random() < 0.4:
        overlay = Image.new('L', (w,h), color=255)
        d2 = ImageDraw.Draw(overlay)
        d2.text((x+random.randint(-10,10), y+random.randint(-6,6)), "overlap", font=font, fill=0)
        arr = np.minimum(arr, np.array(overlay))
    return arr

def generate_dataset(out_dir='data/synth', n=200):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        t = f"Line {i} sample"
        im = random_text_image(t)
        cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), im)
