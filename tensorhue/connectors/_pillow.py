import numpy as np


def _tensorhue_to_numpy_pillow(image, thumbnail, max_size) -> np.ndarray:
    try:
        image = image.convert("RGB")
    except Exception as e:
        raise ValueError("Could not convert image from mode '{mode}' to 'RGB'.") from e

    if thumbnail:
        image.thumbnail(max_size)

    array = np.array(image)
    assert array.dtype == "uint8"

    return array
