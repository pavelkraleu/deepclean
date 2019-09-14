import keras
import os
import pickle
from deepclean import settings
from deepclean.detector_net.detector_data_generator import DetectorDataGenerator
from filelock import FileLock


class BaseSequenceGenerator(keras.utils.Sequence):
    input_layer_name = settings.DETECTOR_INPUT_LAYER_NAME[
        settings.DETECTOR_NETWORK_BACKEND]

    def __init__(self, validation=False):
        self.validation = validation
        self.data_generator = DetectorDataGenerator(validation=validation)

    def __len__(self):
        if self.validation:
            return settings.DETECTOR_BATCHES_PER_EPOCH_VALIDATION
        else:
            return settings.DETECTOR_BATCHES_PER_EPOCH

    def __getitem__(self, index):
        with_graffiti, mask, background = self.get_sample(index)
        return self.process_data(with_graffiti, mask, background)

    def process_data(self, with_graffiti, mask, background):
        raise NotImplementedError()

    def get_sample(self, index):
        cache_file = f"./data_cache/{self.validation}_{index}.p"

        if os.path.exists(cache_file):
            # try:
            #     img_with_graffiti, mask, background = pickle.load(
            #         open(cache_file, "rb"))
            #     return img_with_graffiti, mask, background
            # except Exception:
            #     pass
            img_with_graffiti, mask, background = pickle.load(
                open(cache_file, "rb"))
            return img_with_graffiti, mask, background

        # print(f"Generating {index}")
        with FileLock(cache_file + ".lock", timeout=90):
            img_with_graffiti, mask, background = next(
                self.data_generator.get_sample())
            pickle.dump([img_with_graffiti, mask, background],
                        open(cache_file, "wb"))
            # print(f"Generating {index} done")
            return img_with_graffiti, mask, background

        # with FileLock(cache_file + ".lock", timeout=30):
        #     if os.path.exists(cache_file):
        #         try:
        #             img_with_graffiti, mask, background = pickle.load(
        #                 open(cache_file, "rb"))
        #             return img_with_graffiti, mask, background
        #         except Exception:
        #             pass
        #
        #     # print(f"Generating {index}")
        #     img_with_graffiti, mask, background = next(
        #         self.data_generator.get_sample())
        #     pickle.dump([img_with_graffiti, mask, background],
        #                 open(cache_file, "wb"))
        #     return img_with_graffiti, mask, background
