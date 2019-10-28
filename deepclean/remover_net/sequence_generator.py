import cv2
import skimage
import numpy as np
from deepclean import settings
from deepclean.training_data_generator.base_data_generator import BaseDataGenerator


class SequenceGenerator(BaseDataGenerator):
    """
    Sequence generator fro remover Net
    """
    input_layer_name = settings.DETECTOR_INPUT_LAYER_NAME[
        settings.DETECTOR_NETWORK_BACKEND]

    def __getitem__(self, index):
        batch = self.batch_data[index]
        inputs_img, inputs_mask, outputs_img = self.prepare_batch(batch)

        return ({
            "inputs_img": np.stack(inputs_img) / 255,
            'inputs_mask': 1 - (np.stack(inputs_mask) / 255)
        }, {
            "outputs_img": np.stack(outputs_img) / 255
        })

    def prepare_batch(self, batch):
        inputs_img = []
        inputs_mask = []
        outputs_img = []

        for graffiti_removed, mask_file, background_file in batch:
            graffiti_removed = skimage.io.imread(str(graffiti_removed))
            mask = skimage.io.imread(str(mask_file), as_gray=True) * 255
            background = skimage.io.imread(str(background_file))

            kernel = np.ones((10, 10), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            mask = mask[..., np.newaxis]
            mask = np.repeat(mask, 3, axis=2).astype(int)

            inputs_img.append(graffiti_removed)
            inputs_mask.append(mask)
            outputs_img.append(background)

        return [
            np.array(inputs_img),
            np.array(inputs_mask),
            np.array(outputs_img)
        ]

    def prepare_batches(self, file_ids):
        batch = []

        for file_id in file_ids:
            graffiti_removed = self.data_dir / "graffiti_removed" / f"{file_id}.jpg"
            mask_file = self.data_dir / "masks" / f"{file_id}.jpg"
            background_file = self.data_dir / "background" / f"{file_id}.jpg"

            if graffiti_removed.exists() and mask_file.exists(
            ) and background_file.exists():
                batch.append([graffiti_removed, mask_file, background_file])

            if len(batch) == settings.REMOVER_BATCH_SIZE:
                self.batch_data.append(batch)
                batch = []
