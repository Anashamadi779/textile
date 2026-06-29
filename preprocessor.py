import cv2
import numpy as np
from PIL import Image, ImageEnhance


def preprocess(frame: np.ndarray, target_width: int = 640) -> tuple[np.ndarray, Image.Image]:
    """Resize to target_width preserving aspect ratio, then apply contrast/sharpness boost.

    Returns (display_frame_bgr, pil_image_for_inference).
    """
    h, w = frame.shape[:2]
    new_h = int(h * target_width / w)
    resized = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.2)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)

    display_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return display_frame, pil_img


def preprocess_pil(image: Image.Image, target_width: int = 640) -> Image.Image:
    """Same resize + enhance pipeline for PIL images (used by the Streamlit app)."""
    w, h = image.size
    new_h = int(h * target_width / w)
    image = image.resize((target_width, new_h), Image.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(1.2)
    image = ImageEnhance.Sharpness(image).enhance(1.5)
    return image
