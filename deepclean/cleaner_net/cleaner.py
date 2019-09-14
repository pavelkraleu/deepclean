import glob

import cv2
import imageio
from segmentation_models.backbones import get_preprocessing
import numpy as np
from deepclean import settings
from deepclean.remover_net.network import RemoverNeuralNetwork
from deepclean.detector_net.network import SegmentationNeuralNetwork
from deepclean.remover_net.remover_sequence_generator import RemoverSequenceGenerator
from deepclean.tools import debug_data_generator
import os
import matplotlib.pyplot as plt


class Cleaner:
    video_files = glob.glob('./video_files/*.MOV')

    video_resolution = (1920, 1088)

    remvover_sequence_generator = RemoverSequenceGenerator()

    def __init__(self):

        self.segmentation_net = SegmentationNeuralNetwork()
        self.remover_net = RemoverNeuralNetwork()

    def process_frame(self, original_frame):

        # frame = cv2.imread(frame_path)
        frame = original_frame.copy()
        frame = cv2.resize(frame, settings.DETECTOR_NETWORK_RESOLUTION)

        cv2.imwrite("frame.png", frame)

        segmentation = get_preprocessing(
            settings.DETECTOR_NETWORK_BACKEND)(frame)

        segmentation = self.segmentation_net.model.predict([[segmentation]])[0]

        # cv2.imwrite("segmentation.png", segmentation)

        ret, segmentation = cv2.threshold(segmentation, 0.5, 1.0,
                                          cv2.THRESH_BINARY)

        # segmentation = 1 - segmentation
        #
        # print(segmentation.min())
        # print(segmentation.max())

        # cv2.imwrite("segmentation-a.png", segmentation * 255)

        data_to_remove = self.remvover_sequence_generator.process_data(
            np.array([frame]), np.array([segmentation * 255]),
            np.array([frame]))

        # debug_data_generator([data_to_remove])

        removed_frame = self.remover_net.model.predict(data_to_remove[0])[0]

        removed_frame = removed_frame * 255
        # cv2.imwrite("out.png", removed_frame)

        mask = 1 - data_to_remove[0]['inputs_mask'][0]
        mask = mask.astype(bool)

        frame = original_frame.copy()
        frame = cv2.resize(frame, settings.DETECTOR_NETWORK_RESOLUTION)
        frame[mask] = removed_frame[mask]

        cv2.imwrite("frame.png", frame)

        return frame

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

    def process_video(self):
        for video_file in self.video_files:
            print(video_file)
            output_video_file = f'./training_results/clean_{os.path.basename(video_file)}.mp4'
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

                    result = self.process_frame(frame_orig)

                    self.save_sample(frame_orig, result,
                                     f'./training_results/tmp.png')

                    # dddd

                    img_file = cv2.cvtColor(
                        cv2.imread(f'./training_results/tmp.png'),
                        cv2.COLOR_BGR2RGB)
                    img_file = cv2.resize(img_file, self.video_resolution)
                    writer.append_data(img_file)
