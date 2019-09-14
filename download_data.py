import pandas as pd
import sys
from skimage import io
import json
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
"""Script to convert LabelBox CSV to Pandas DF"""

pd.set_option('display.max_colwidth', 1000)

input_csv_file = sys.argv[1]


def process_labels(labels):

    if labels.get('colors_used', 'single_color') == 'single_color':
        colors_used = 1
    else:
        # More than one color used
        colors_used = 0

    return labels.get('graffiti_type', 'tag'), \
           colors_used, \
           labels.get('readable_text', ''), \
           labels.get('tool_used', 'marker')


def process_masks(labels, image_shape):

    if 'Background Graffiti' in labels['segmentationMasksByName']:
        background_graffiti = cv2.bitwise_not(
            cv2.inRange(
                io.imread(
                    labels['segmentationMasksByName']['Background Graffiti']),
                0, 0))
    else:
        background_graffiti = np.zeros(image_shape[0:2], dtype=np.uint8)

    return cv2.bitwise_not(cv2.inRange(io.imread(labels['segmentationMasksByName']['Background']), 0, 0)),\
           cv2.bitwise_not(cv2.inRange(io.imread(labels['segmentationMasksByName']['Graffiti']), 0, 0)), \
           cv2.bitwise_not(cv2.inRange(io.imread(labels['segmentationMasksByName']['Incomplete Graffiti']), 0, 0)), \
           background_graffiti


def check_pickle_file(pickle_file, output_dir):

    sample = pickle.load(open(pickle_file, 'rb'))

    img = np.array(sample['image'])
    mask = np.array(sample['graffiti_mask'])

    print(img.shape)
    print()

    blank_image = np.zeros((1024, 1024, 3))
    mask_image = np.zeros((1024, 1024))

    if img.shape[0] > img.shape[1]:
        y_offset = int((1024 - img.shape[1]) / 2)
        x_offset = 0
    else:
        y_offset = 0
        x_offset = int((1024 - img.shape[0]) / 2)

    y_offset = x_offset = 0

    print(x_offset, y_offset)

    blank_image[y_offset:y_offset + img.shape[0], x_offset:x_offset +
                img.shape[1]] = img
    mask_image[y_offset:y_offset + mask.shape[0], x_offset:x_offset +
               mask.shape[1]] = mask

    cv2.imwrite(
        f'data/{output_dir}_images/{output_dir}/{sample["id"]}_image.png',
        cv2.resize(blank_image, (512, 512)))
    cv2.imwrite(
        f'data/{output_dir}_masks/{output_dir}/{sample["id"]}_graffiti_mask.png',
        cv2.resize(mask_image, (512, 512)))


# for col in list(df):

# print(df.head()[col])
# print()

# original_datset_df = pd.read_csv(dataset_csv,
#                                  index_col='hash_average',
#                                  usecols=[
#                                      'gps_latitude', 'gps_longitude',
#                                      'hash_average', 'original_file_name'
#                                  ])
#
# print(original_datset_df)


def process_df(df, output_dir):

    for index, row in df.iterrows():

        data_sample = {}

        try:
            labels = json.loads(row['Label'])
        except json.decoder.JSONDecodeError:
            continue

        image = io.imread(row['Labeled Data'])

        # print(labels)

        graffiti_type, colors_used, readable_text, tool_used = process_labels(
            labels)

        file_id = row['External ID'].split('.')[0]

        print(file_id)

        data_sample['id'] = file_id

        data_sample['image'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data_sample['graffiti_type'] = graffiti_type
        data_sample['colors_used'] = colors_used
        data_sample['readable_text'] = readable_text
        data_sample['tool_used'] = tool_used

        background_mask, graffiti_mask, incomplete_graffiti_mask, background_graffiti_mask = process_masks(
            labels, image.shape)

        data_sample['background_mask'] = background_mask
        data_sample['graffiti_mask'] = graffiti_mask
        data_sample['incomplete_graffiti_mask'] = incomplete_graffiti_mask
        data_sample['background_graffiti_mask'] = background_graffiti_mask

        # print(data_sample)

        pickle.dump(data_sample, open(f'tmp.p', 'wb'))

        check_pickle_file(f'tmp.p', output_dir)


df = pd.read_csv(input_csv_file)

train, test = train_test_split(df, test_size=0.2)

print(len(train))

print(len(test))

process_df(train, 'train')
process_df(test, 'test')
