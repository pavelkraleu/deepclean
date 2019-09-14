import cv2
import numpy as np
from segmentation_models.backbones import get_preprocessing
from deepclean import settings
from deepclean.data_generator.base_sequence_generator import BaseSequenceGenerator


class RemoverSequenceGenerator(BaseSequenceGenerator):
    def process_data(self, with_graffiti, masks, backgrounds):

        final_masks = []
        final_backgrounds = []
        final_backgrounds_orig = []

        for i in range(masks.shape[0]):

            mask = masks[i]
            background = backgrounds[i]
            background_orig = np.array(backgrounds[i])

            kernel = np.ones((10, 10), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            mask = mask[..., np.newaxis]
            # print(mask.shape)

            mask = np.repeat(mask, 3, axis=2).astype(int)
            background[mask == 255] = 255

            final_masks.append(mask)
            final_backgrounds.append(background)
            final_backgrounds_orig.append(background_orig)

            # output_data = mask / 255

        return ({
            "inputs_img": np.stack(final_backgrounds) / 255,
            'inputs_mask': 1 - (np.stack(final_masks) / 255)
        }, {
            "outputs_img": np.stack(final_backgrounds_orig) / 255
        })
