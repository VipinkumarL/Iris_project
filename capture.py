import base64
import os
import tempfile
from uuid import uuid4


def _split_data_url(image_data):
    if not image_data or "," not in image_data:
        return None, None

    return image_data.split(",", 1)


def _decode_image_bytes(encoded_image):
    try:
        return base64.b64decode(encoded_image, validate=True)
    except (ValueError, TypeError):
        return None


def save_image_from_data_url(image_data, output_dir=None):
    header, encoded_image = _split_data_url(image_data)
    if not header or not encoded_image:
        return None

    if "image/" not in header:
        return None

    raw_bytes = _decode_image_bytes(encoded_image)
    if raw_bytes is None:
        return None

    target_dir = output_dir or tempfile.gettempdir()
    os.makedirs(target_dir, exist_ok=True)

    image_path = os.path.join(target_dir, f"iris_capture_{uuid4().hex}.jpg")
    with open(image_path, "wb") as image_file:
        image_file.write(raw_bytes)

    return image_path
