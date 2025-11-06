# synthetic/synth_generator.py
import os, cv2, numpy as np, random
from PIL import Image, ImageDraw, ImageFont

def random_text_image(text, w=512, h=256, font_size=32, fade_prob=0.6):
    im = Image.new('L', (w,h), color=255)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    y = random.randint(5, max(5,h-font_size-5))
    draw.text((10,y), text, font=font, fill=0)
    arr = np.array(im).astype(np.uint8)
    if random.random() < fade_prob:
        mask = np.linspace(1,0.3,w).astype(np.float32)
        arr = (arr * mask[np.newaxis,:]).astype(np.uint8)
    if random.random() < 0.4:
        # overlap
        overlay = Image.new('L',(w,h),255)
        d2 = ImageDraw.Draw(overlay)
        d2.text((10+random.randint(-15,15), y+random.randint(-8,8)), "overlap", font=font, fill=0)
        arr = np.minimum(arr, np.array(overlay))
    return arr

def generate(out_dir='data/synth', n=500):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        t = f"Sample {i}"
        im = random_text_image(t)
        cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), im)

if __name__ == "__main__":
    generate()
