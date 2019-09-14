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


class SegmentationModelDebugger(keras.callbacks.Callback):

    video_resolution = (1920, 1088)

    video_files = glob.glob('./video_files/*.MOV')

    def __init__(self, trainer):
        super().__init__()

        self.trainer = trainer

    def on_epoch_end(self, epoch, logs={}):

        if epoch % 4 == 0 and epoch != 0:
            self.debug_validation_data(epoch)

    def debug_video_data(self, epoch):
        for video_file in self.video_files:
            print(video_file)
            output_video_file = f'./training_results/e{epoch}_{os.path.basename(video_file)}.mp4'
            with imageio.get_writer(output_video_file, mode='?',
                                    fps=30) as writer:

                capture = cv2.VideoCapture(video_file)
                while (capture.isOpened()):
                    ret, frame = capture.read()

                    if not ret:
                        break

                    frame_orig = cv2.cvtColor(
                        cv2.resize(frame,
                                   (settings.DETECTOR_NETWORK_RESOLUTION)),
                        cv2.COLOR_BGR2RGB)

                    frame = cv2.resize(frame,
                                       (settings.DETECTOR_NETWORK_RESOLUTION))
                    frame = get_preprocessing(
                        settings.DETECTOR_NETWORK_BACKEND)(frame)
                    result = self.trainer.model.predict([[frame]])

                    self.save_sample(frame_orig,
                                     np.squeeze(result) * 255,
                                     f'./training_results/tmp.png')

                    img_file = cv2.cvtColor(
                        cv2.imread(f'./training_results/tmp.png'),
                        cv2.COLOR_BGR2RGB)
                    img_file = cv2.resize(img_file, self.video_resolution)
                    writer.append_data(img_file)

    # def debug_single_image(self, jpg_img_path, epoch, output_path=None):
    #     img_file = cv2.cvtColor(cv2.imread(jpg_img_path), cv2.COLOR_BGR2RGB)
    #     img_orig = cv2.resize(img_file, self.trainer.nn_input_shape)
    #     img = get_preprocessing(self.trainer.backbone)(img_orig)
    #
    #     result = np.squeeze(self.trainer.model.predict(np.array([img]))[0])
    #
    #     if output_path is None:
    #         output_path = f'./training_results_{self.trainer.backbone}/{epoch}/test/{os.path.basename(jpg_img_path)}'
    #
    #     self.save_sample(img_file, result * 255, output_path)

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

            if bi % 5 != 0:
                continue

            result = self.trainer.model.predict(
                batch[0][settings.DETECTOR_INPUT_LAYER_NAME[
                    settings.DETECTOR_NETWORK_BACKEND]])

            for i, sample in enumerate(result):
                if i % 2 != 0:
                    continue

                print(f'./training_results/e{epoch}_b{bi}_s{i}.jpg')
                self.save_sample(
                    cv2.cvtColor(batch[0]['raw_data'][i], cv2.COLOR_BGR2RGB),
                    np.squeeze(sample) * 255,
                    f'./training_results/e{epoch}_b{bi}_s{i}.jpg')
