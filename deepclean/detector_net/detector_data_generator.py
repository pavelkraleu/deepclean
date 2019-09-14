import glob
import random

import cv2
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from deepclean import settings
from deepclean.tools import get_random_background
import imgaug.augmenters as iaa


class DetectorDataGenerator:
    train_datagen = ImageDataGenerator()

    negative_images = glob.glob('./background_images/*.JPG')

    image_agumentations = iaa.Sequential(
        [
            # iaa.ElasticTransformation(alpha=100, sigma=30),
            # iaa.Multiply((0.5, 1.5), per_channel=0.5),
            # iaa.Affine(translate_px={
            #     "x": (-128, 128),
            #     "y": (-128, 128)
            # }),
            # iaa.PiecewiseAffine(scale=(0.01, 0.05)),
            iaa.Affine(rotate=(-30, 30)),
            # iaa.Resize((0.3, 7.0)),
            # iaa.GaussianBlur(sigma=(0.0, 3.0)),
            # iaa.Sometimes(0.5,
            #               iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))),
            # iaa.Affine(cval=(0, 255)),
            # iaa.Multiply((0.5, 1.5)),
            # iaa.Sometimes(0.5, iaa.MultiplyElementwise((0.5, 1.5))),
            # iaa.Multiply((0.5, 1.5), per_channel=0.5),
            # iaa.ContrastNormalization((0.5, 1.5))
        ],
        random_order=True)

    def __init__(self, validation=False):
        if validation:
            train_img_dir = 'data/test_images/'
            train_mask_dir = 'data/test_masks/'
        else:
            train_img_dir = 'data/train_images/'
            train_mask_dir = 'data/train_masks/'

        train_image_generator = self.train_datagen.flow_from_directory(
            train_img_dir,
            batch_size=settings.DETECTOR_BATCH_SIZE,
            target_size=settings.DETECTOR_NETWORK_RESOLUTION,
            class_mode=None,
            seed=10)

        train_mask_generator = self.train_datagen.flow_from_directory(
            train_mask_dir,
            batch_size=settings.DETECTOR_BATCH_SIZE,
            target_size=settings.DETECTOR_NETWORK_RESOLUTION,
            class_mode=None,
            seed=10)

        self.train_generator = zip(train_image_generator, train_mask_generator)

    def get_sample(self):

        images, masks = next(self.train_generator)

        masks = [
            cv2.inRange(255 - np.array(mask).astype(np.uint8), 0, 0)
            for mask in masks
        ]

        images = [np.array(image).astype(np.uint8) for image in images]

        # print('agumenting')
        images, masks = self.image_agumentations(images=images,
                                                 segmentation_maps=masks)

        # print('agumenting done')
        final_images = []
        final_maps = []
        final_original = []

        for i in range(len(images)):

            image = np.array(images[i])
            mask = np.array(masks[i])

            image = cv2.resize(image, settings.DETECTOR_NETWORK_RESOLUTION)
            mask = cv2.resize(mask, settings.DETECTOR_NETWORK_RESOLUTION)

            random_background = get_random_background(
                *settings.DETECTOR_NETWORK_RESOLUTION, self.negative_images)

            image_on_background = self.paste_on_background(
                image, mask, random_background)

            final_images.append(image_on_background)
            final_maps.append(mask)
            final_original.append(random_background)

            # print('writing')
            rnd = random.randint(1, 30)
            cv2.imwrite(f"img-img_{rnd}.png", image_on_background)
            cv2.imwrite(f"img-mask_{rnd}.png", mask)
            cv2.imwrite(f"img-bg_{rnd}.png", random_background)

        yield np.stack(final_images), np.stack(final_maps), np.stack(
            final_original)

    def paste_on_background(self, image, mask, background_image):

        foreground = image.astype(float)
        background = background_image.astype(float)

        alpha = cv2.inRange(mask, 0, 0)
        alpha = cv2.bitwise_not(alpha).astype(float) / 255

        alpha = np.stack((alpha, ) * 3, axis=-1)

        foreground = cv2.multiply(alpha, foreground)

        background = cv2.multiply(1.0 - alpha, background)

        return cv2.add(foreground, background).astype(np.uint8)
