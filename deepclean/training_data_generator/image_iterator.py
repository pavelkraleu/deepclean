from pathlib import PurePath
import skimage
from deepclean import settings
import numpy as np


def get_image_ids(image_directory):
    images = image_directory.glob("*.png")
    return [image.name.split(".")[0] for image in images]


def image_iterator(validation=False):
    """Endless iterator generating image and corresponding mask"""

    data_directory = settings.IMAGE_TRAINING_DATA_DIR
    if validation:
        data_directory = settings.IMAGE_VALIDATION_DATA_DIR

    image_ids = get_image_ids(data_directory / "image")

    while True:
        for image_id in image_ids:
            image_path = PurePath(data_directory, "image", image_id + ".png")
            mask_path = PurePath(data_directory, "mask", image_id + ".png")
            image = skimage.io.imread(str(image_path))
            mask = skimage.io.imread(str(mask_path), as_gray=True)
            yield image.astype(np.uint8), mask.astype(np.uint8)
