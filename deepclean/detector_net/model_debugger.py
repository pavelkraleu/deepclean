import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from deepclean import settings


class SegmentationModelDebugger(keras.callbacks.Callback):
    """
    This callback renders validation samples during training of detector network
    """
    def __init__(self, trainer, decoder_training):
        super().__init__()
        self.decoder_training_suffix = "decoder_" if decoder_training else ""
        self.trainer = trainer

    def on_epoch_end(self, epoch, logs={}):
        for bi, batch in enumerate(self.trainer.validation_data):
            if bi % 100 != 0:
                continue
            result = self.trainer.model.predict(
                batch[0][settings.DETECTOR_INPUT_LAYER_NAME[
                    settings.DETECTOR_NETWORK_BACKEND]])
            for i, sample in enumerate(result):
                if i % 2 != 0:
                    continue
                self.save_sample(
                    cv2.cvtColor(batch[0]['raw_data'][i], cv2.COLOR_BGR2RGB),
                    np.squeeze(sample) * 255,
                    f'./training_results_{settings.DETECTOR_NETWORK_BACKEND}/{self.decoder_training_suffix}e{epoch}_b{bi}_s{i}.jpg'
                )

    def save_sample(self, x, y, output_path):
        fig, axs = plt.subplots(1, 2, figsize=(16, 9), dpi=150)
        axs[0].imshow(x)
        axs[1].imshow(y)

        for i in [0, 1]:
            axs[i].tick_params(left=False, bottom=False)
            axs[i].set_yticklabels([])
            axs[i].set_xticklabels([])

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
