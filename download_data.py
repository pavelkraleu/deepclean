import pandas as pd
import sys
from skimage import io
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os

from deepclean import settings

input_csv_file = sys.argv[1]


def process_image(image):
    """Convert a single image to 1024x1024 image"""

    blank_shape = (1024, 1024, 3)
    if image.ndim == 2:
        blank_shape = blank_shape[0:2]

    blank_image = np.zeros(blank_shape, dtype=np.uint8)
    y_offset = x_offset = 0

    blank_image[y_offset:y_offset + image.shape[0], x_offset:x_offset +
                image.shape[1]] = image

    return cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)


def save_image(image, image_type, destination_dir, file_id):
    output_path = f"./data/{destination_dir}/{image_type}"
    os.makedirs(output_path, exist_ok=True)

    cv2.imwrite(output_path + f"/{file_id}.png",
                cv2.resize(image, settings.DETECTOR_NETWORK_RESOLUTION))


def process_df(df, destination_dir):
    """Process Pandas DF and prepares images for training NN"""
    for index, row in df.iterrows():

        try:
            labels = json.loads(row['Label'])
        except json.decoder.JSONDecodeError:
            continue

        file_id = row['External ID'].split('.')[0]
        print(f"Processing ID {file_id}")

        image = io.imread(row['Labeled Data'])
        graffiti_mask = cv2.bitwise_not(
            cv2.inRange(
                io.imread(labels['segmentationMasksByName']['Graffiti']), 0,
                0))

        square_image = process_image(image)
        square_mask = process_image(graffiti_mask)

        save_image(square_image, "image", destination_dir, file_id)
        save_image(square_mask, "mask", destination_dir, file_id)


if __name__ == "__main__":
    df = pd.read_csv(input_csv_file)
    train, test = train_test_split(df, test_size=0.2)
    process_df(train, 'train')
    process_df(test, 'test')
