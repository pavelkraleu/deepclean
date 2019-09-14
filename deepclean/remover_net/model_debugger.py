import glob
import os
import pickle
import shutil

import imageio
import keras
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import Adam
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import numpy as np
import cv2
from segmentation_models.utils import set_trainable
import matplotlib.pyplot as plt
from skimage import transform

from deepclean import settings


class RemoverModelDebugger(keras.callbacks.Callback):

    video_resolution = (1920, 1088)

    video_files = glob.glob('./video_files/*.MOV')

    def __init__(self, trainer):
        super().__init__()

        self.trainer = trainer

    def on_epoch_end(self, epoch, logs={}):

        self.debug_validation_data(epoch)

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

    def debug_validation_data(self, epoch):

        for bi, batch in enumerate(self.trainer.validation_data):

            result = self.trainer.model.model.predict(batch[0])

            for i, sample in enumerate(result):

                print(sample.shape)

                print(f'./training_results/e{epoch}_b{bi}_s{i}.jpg')
                self.save_sample(
                    batch[0]['inputs_img'][i],
                    # batch[0]['inputs_mask'][i],
                    sample,
                    f'./training_results/e{epoch}_b{bi}_s{i}.jpg')
