import keras
from deepclean import settings


class BaseDataGenerator(keras.utils.Sequence):
    def __init__(self, validation=False):
        self.validation = validation
        self.batch_data = []

        self.data_dir = settings.PREPARED_VALIDATION_DATA_DIR if validation else settings.PREPARED_TRAINING_DATA_DIR

        file_ids = self.get_file_ids()
        self.prepare_batches(file_ids)

    def __len__(self):
        return len(self.batch_data)

    def get_file_ids(self):
        images_dir = self.data_dir / "graffiti"
        all_images = images_dir.glob("*.jpg")
        return [image.name.split(".")[0] for image in all_images]

    def prepare_batch(self, batch):
        """
        This method should return one batch from self.batch_data
        """
        raise NotImplementedError()

    def prepare_batches(self, file_ids):
        """
        This method should fill self.batch_data, so each entry contains only one batch
        """
        raise NotImplementedError()