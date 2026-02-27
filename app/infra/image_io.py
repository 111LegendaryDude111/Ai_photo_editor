import base64
from io import BytesIO

from PIL import Image


def decode_base64_image(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def encode_base64_image(image: Image.Image, format_name: str = "PNG") -> str:
    buf = BytesIO()
    image.save(buf, format=format_name)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_to_bytes(image: Image.Image, format_name: str = "PNG") -> bytes:
    buf = BytesIO()
    image.save(buf, format=format_name)
    return buf.getvalue()
