import os
import random

import cv2
import skimage

from deepclean import settings
from deepclean.tools import paste_on_background
from deepclean.training_data_generator.image_iterator import image_iterator
import imgaug.augmenters as iaa
import numpy as np

# Augmentations applied to single graffiti samples later copied to a new background
graffiti_image_augmentations = iaa.Sequential([
    iaa.ElasticTransformation(alpha=100, sigma=30),
    iaa.Add((-40, 40), per_channel=0.5),
    iaa.Affine(shear=(-16, 16)),
    iaa.Affine(scale=(0.3, 0.7)),
    iaa.Affine(rotate=(-30, 30)),
    iaa.Affine(translate_px={
        "x": (-200, 200),
        "y": (-200, 200)
    }),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.GaussianBlur(sigma=(0.0, 1.0)),
])

# Augmentations applied to a random background to be used for new graffiti sample
background_image_augmentations = iaa.Sequential([
    iaa.ElasticTransformation(alpha=100, sigma=30),
    iaa.GaussianBlur(sigma=(0.0, 2.0)),
    iaa.Affine(shear=(-3, 3)),
    iaa.Affine(rotate=(-3, 3)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Add((-40, 40), per_channel=0.5)
])

# Augmentations applied to a final sample with graffiti on new background
final_image_augmentations = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0.0, 1.0)),
])


def gen_processor_counter():
    images_stack = np.zeros(settings.DETECTOR_NETWORK_RESOLUTION + (3, ),
                            dtype=np.uint8)
    masks_stack = np.zeros(settings.DETECTOR_NETWORK_RESOLUTION,
                           dtype=np.uint8)
    images_count = 0

    return images_stack, masks_stack, images_count


def get_random_background(width, height, negative_images):

    random.shuffle(negative_images)
    img = cv2.imread(negative_images[0])
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    [img] = background_image_augmentations(images=[img])

    min_value_img = min(img.shape[0:2])
    min_value_size = min((width, height))
    min_ratio = min_value_size / min_value_img
    img_resize_ratio = random.uniform(min_ratio, 1.0)

    img = skimage.transform.rescale(img,
                                    img_resize_ratio,
                                    multichannel=True,
                                    preserve_range=True).astype(np.uint8)

    if min(img.shape[:2]) < min(width, height):
        # Image is smaller than self.target_size. Scale it up
        resize_ratio = min(width, height) / min(img.shape[:2])

    else:
        # Image is larger than self.target_size. Scale it down a little
        resize_ratio = min(width, height) / min(img.shape[:2])
        resize_ratio = random.uniform(resize_ratio, 1.0)

    img = skimage.transform.rescale(img,
                                    resize_ratio,
                                    multichannel=True,
                                    anti_aliasing=True,
                                    mode='constant',
                                    preserve_range=True)

    y_range = img.shape[0] - height
    x_range = img.shape[1] - width

    y_skip = random.randint(0, y_range)
    x_skip = random.randint(0, x_range)

    crop_img = img[y_skip:y_skip + height, x_skip:x_skip + width]
    return crop_img


def save_samples(mask, image_on_background, background_removed,
                 original_background):

    sample_id = random.randint(0, 2**50)

    def save(sample_dir, image):
        os.makedirs(sample_dir, exist_ok=True)
        skimage.io.imsave(f"{sample_dir}/{sample_id}.jpg",
                          image.astype(np.uint8),
                          check_contrast=False)

    save(f"./training_data/masks/", mask)
    save(f"./training_data/graffiti/", image_on_background)
    save(f"./training_data/graffiti_removed/", background_removed)
    save(f"./training_data/background/", original_background)


def image_processor(number_images_generate, validation=False):

    image_generator = image_iterator(validation=validation)

    for _ in range(number_images_generate):
        images_to_stack = random.choices([1, 2, 3, 4], [0.7, 0.1, 0.1, 0.1])[0]
        print(images_to_stack)

        images_stack, masks_stack, images_count = gen_processor_counter()

        for _ in range(images_to_stack):

            image, mask = next(image_generator)

            [image
             ], [mask] = graffiti_image_augmentations(images=[image],
                                                      segmentation_maps=[mask])
            image = skimage.transform.resize(
                image,
                settings.DETECTOR_NETWORK_RESOLUTION,
                preserve_range=True).astype(np.uint8)
            mask = skimage.transform.resize(
                mask,
                settings.DETECTOR_NETWORK_RESOLUTION,
                preserve_range=True).astype(np.uint8)

            graffiti_collision = cv2.inRange(
                cv2.bitwise_and(images_stack, image), 0, 0)

            image[graffiti_collision == 0] = 0
            mask[graffiti_collision == 0] = 0

            images_stack += image
            masks_stack += mask
            images_count += 1

        random_background = get_random_background(
            *settings.DETECTOR_NETWORK_RESOLUTION, settings.NEGATIVE_IMAGES)
        image_on_background = paste_on_background(images_stack, masks_stack,
                                                  random_background)

        [image_on_background
         ], [masks_stack
             ] = final_image_augmentations(images=[image_on_background],
                                           segmentation_maps=[masks_stack])

        masks_stack *= 255

        background_removed = np.array(random_background)
        masks_stack = np.repeat(masks_stack[..., np.newaxis], 3,
                                axis=2).astype(np.uint8)
        background_removed[masks_stack == 255] = 255

        save_samples(masks_stack, image_on_background, background_removed,
                     random_background)
