import skimage
from segmentation_models.backbones import get_preprocessing
import numpy as np
from deepclean import settings
from deepclean.training_data_generator.base_data_generator import BaseDataGenerator


class SequenceGenerator(BaseDataGenerator):
    input_layer_name = settings.DETECTOR_INPUT_LAYER_NAME[
        settings.DETECTOR_NETWORK_BACKEND]

    def __getitem__(self, index):
        batch = self.batch_data[index]
        images, images_processed, masks = self.prepare_batch(batch)

        return ({
            self.input_layer_name: images_processed,
            "raw_data": images
        }, {
            'sigmoid': masks[..., np.newaxis]
        })

    def prepare_batch(self, batch):

        images = []
        images_processed = []
        masks = []

        for graffiti_file, mask_file in batch:
            image = skimage.io.imread(str(graffiti_file))
            mask = skimage.io.imread(str(mask_file), as_gray=True)

            # TODO this should be in constructor
            image_processed = get_preprocessing(
                settings.DETECTOR_NETWORK_BACKEND)(image)

            images.append(image)
            masks.append(mask)
            images_processed.append(image_processed)

        return [np.array(images), np.array(images_processed), np.array(masks)]

    def prepare_batches(self, file_ids):
        batch = []

        for file_id in file_ids:
            graffiti_file = self.data_dir / "graffiti" / f"{file_id}.jpg"
            mask_file = self.data_dir / "masks" / f"{file_id}.jpg"

            if graffiti_file.exists() and mask_file.exists():
                batch.append([graffiti_file, mask_file])

            if len(batch) == settings.DETECTOR_BATCH_SIZE:
                self.batch_data.append(batch)
                batch = []
