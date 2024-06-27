import os
from PIL import Image
import pytest
from tensorhue.connectors._pillow import _tensorhue_to_numpy_pillow


@pytest.mark.parametrize("thumbnail", [True, False])
def test_image_modes(thumbnail):
    "Tests .jpg, .png and .gif images in different color modes (RGB, L, RGBA, etc.)"
    image_dir = "./tests/test_resources/"
    for file in os.listdir(image_dir):
        img = Image.open(image_dir + file)
        array = _tensorhue_to_numpy_pillow(img, thumbnail=thumbnail, max_size=(100, 138))
        assert array.shape == ((100, 100, 3) if thumbnail else (600, 600, 3))
        if img.getbands()[-1] == "A":
            assert array[0, 0, 0] == 0  # top left pixel is black (PIL default: transparent -> black)
        else:
            assert array[0, 0, 0] == 255  # top left pixel is white
