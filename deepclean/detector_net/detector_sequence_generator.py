import cv2
import numpy as np
from segmentation_models.backbones import get_preprocessing
from deepclean import settings
from deepclean.data_generator.base_sequence_generator import BaseSequenceGenerator
import imgaug.augmenters as iaa


class DetectorSequenceGenerator(BaseSequenceGenerator):

    image_agumentations = iaa.Sequential(
        [
            iaa.ElasticTransformation(alpha=100, sigma=30),
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.Affine(translate_px={
                "x": (-128, 128),
                "y": (-128, 128)
            }),
            iaa.PiecewiseAffine(scale=(0.01, 0.05)),
            iaa.Resize((0.3, 7.0)),
            iaa.Affine(rotate=(-30, 30)),
            # iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255))),
            iaa.Affine(cval=(0, 255)),
            iaa.Multiply((0.5, 1.2)),
            iaa.Sometimes(0.5, iaa.MultiplyElementwise((0.5, 1.5))),
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
            iaa.ContrastNormalization((0.5, 1.2))
        ],
        random_order=True)

    def process_data(self, with_graffiti, mask, background):

        preprocessing = get_preprocessing(settings.DETECTOR_NETWORK_BACKEND)
        with_graffiti, mask = self.image_agumentations(images=with_graffiti,
                                                       segmentation_maps=mask)

        # with_graffiti = self.image_agumentations(images=with_graffiti)

        output_data = mask / 255

        return ({
            self.input_layer_name: preprocessing(with_graffiti),
            "raw_data": with_graffiti
        }, {
            'sigmoid': output_data[..., np.newaxis]
        })
