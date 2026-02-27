import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw


def image_to_b64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_face_like_image(width: int = 256, height: int = 256, hue_shift: int = 0) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width]
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[..., 0] = (xx + hue_shift) % 255
    arr[..., 1] = (yy * 2 + hue_shift) % 255
    arr[..., 2] = ((xx + yy) // 2 + hue_shift) % 255

    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse((80, 40, 180, 160), outline=(255, 240, 220), width=4)
    draw.ellipse((110, 80, 125, 95), fill=(20, 20, 20))
    draw.ellipse((135, 80, 150, 95), fill=(20, 20, 20))
    draw.arc((115, 105, 145, 130), start=10, end=170, fill=(30, 30, 30), width=3)
    return img
