import glob
import cv2
import imageio
from segmentation_models.backbones import get_preprocessing
import numpy as np
from deepclean import settings
from deepclean.remover_net.network import RemoverNeuralNetwork
from deepclean.detector_net.network import SegmentationNeuralNetwork
import os
import matplotlib.pyplot as plt


class Cleaner:
    """
    Class responsible for processing video files and removing graffiti from them
    """
    video_files = glob.glob(settings.CLEANER_VIDEO_FILES_PATH)

    video_resolution = (1920, 1088)

    def __init__(self):

        self.segmentation_net = SegmentationNeuralNetwork()
        self.remover_net = RemoverNeuralNetwork()

    def process_data(self, with_graffiti, masks, backgrounds):
        """
        Prepares frame for Remover network
        """
        final_masks = []
        final_backgrounds = []
        final_backgrounds_orig = []

        for i in range(masks.shape[0]):

            mask = masks[i]
            background = backgrounds[i]
            background_orig = np.array(backgrounds[i])

            kernel = np.ones((5, 5), np.uint8)
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

    def process_frame(self, original_frame):
        frame = original_frame.copy()
        frame = cv2.resize(frame, settings.DETECTOR_NETWORK_RESOLUTION)

        segmentation = get_preprocessing(
            settings.DETECTOR_NETWORK_BACKEND)(frame)

        segmentation_orig = self.segmentation_net.model.predict(
            [[segmentation]])[0]

        ret, segmentation = cv2.threshold(segmentation_orig,
                                          settings.GENERATE_OUTPUT_THRESHOLD,
                                          1.0, cv2.THRESH_BINARY)

        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(segmentation, cv2.MORPH_OPEN, kernel)
        opening = cv2.dilate(opening, np.ones((3, 3)))

        data_to_remove = self.process_data(np.array([frame]),
                                           np.array([opening * 255]),
                                           np.array([frame]))

        removed_frame = self.remover_net.model.predict(data_to_remove[0])[0]

        removed_frame = removed_frame * 255
        # cv2.imwrite("out.png", removed_frame)

        mask = 1 - data_to_remove[0]['inputs_mask'][0]
        mask = mask.astype(bool)

        frame = original_frame.copy()
        frame = cv2.resize(frame, settings.DETECTOR_NETWORK_RESOLUTION)
        frame[mask] = removed_frame[mask]

        return frame, opening

    def save_sample(self, x, y, mask, output_path):
        """
        Generates a single frame with source and processed frame and also graffiti mask.
        This frame is stored to disk.
        TODO : Is there some way how to move image from matplotlib to imageio without storing it on disk ?
        """
        fig, axs = plt.subplots(1, 2, figsize=(16, 9), dpi=150)
        axs[0].imshow(x)
        axs[1].imshow(y)

        axs[0].set_title(f"Source frame", fontdict={'fontsize': 20})
        axs[1].set_title(f"Processed frame", fontdict={'fontsize': 20})

        for i in [0, 1]:
            axs[i].tick_params(left=False, bottom=False)
            axs[i].set_yticklabels([])
            axs[i].set_xticklabels([])

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

        frame = cv2.imread(output_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * 255
        mask = cv2.resize(mask, (400, 400))

        x_offset = int((frame.shape[1] - mask.shape[1]) / 2)
        y_offset = frame.shape[0] - mask.shape[0] - 50
        frame[y_offset:y_offset + mask.shape[0], x_offset:x_offset +
              mask.shape[1]] = mask

        cv2.imwrite(output_path, frame)

    def process_video(self):
        for video_file in self.video_files:
            print(video_file)
            output_video_file = f'./training_results/clean_{os.path.basename(video_file)}.mp4'
            capture = cv2.VideoCapture(video_file)
            fps = int(capture.get(cv2.CAP_PROP_FPS))
            with imageio.get_writer(output_video_file, mode='?',
                                    fps=fps) as writer:

                while (capture.isOpened()):
                    ret, frame = capture.read()

                    if not ret:
                        break

                    frame_orig = cv2.cvtColor(
                        cv2.resize(frame,
                                   (settings.DETECTOR_NETWORK_RESOLUTION)),
                        cv2.COLOR_BGR2RGB)

                    result, segmentation = self.process_frame(frame_orig)

                    # Save frame to Ramdisk and make SSD happy :)
                    self.save_sample(frame_orig, result, segmentation,
                                     f'./ramdisk/tmp.png')

                    img_file = cv2.cvtColor(cv2.imread(f'./ramdisk/tmp.png'),
                                            cv2.COLOR_BGR2RGB)
                    img_file = cv2.resize(img_file, self.video_resolution)
                    writer.append_data(img_file)
