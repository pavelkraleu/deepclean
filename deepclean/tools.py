import random
import numpy as np
import cv2
import skimage


def get_random_background(width, height, negative_images):
    """
    Makes image with given size which doesn't contain graffiti

    :param width: Resulting width in pixels
    :param height: Resulting height in pixels
    :param negative_images: List of paths to negative images

    """

    random.shuffle(negative_images)
    img = cv2.imread(negative_images[0])

    min_value_img = min(img.shape[0:2])
    min_value_size = min((width, height))
    min_ratio = min_value_size / min_value_img
    img_resize_ratio = random.uniform(min_ratio, 1.0)

    img = skimage.transform.rescale(img,
                                    img_resize_ratio,
                                    multichannel=True,
                                    preserve_range=True).astype(np.uint8)

    if min(img.shape[:2]) < min(width, height):
        # Image is smaller than self.target_size. Scale it up
        resize_ratio = min(width, height) / min(img.shape[:2])

    else:
        # Image is larger than self.target_size. Scale it down a little
        resize_ratio = min(width, height) / min(img.shape[:2])
        resize_ratio = random.uniform(resize_ratio, 1.0)

    img = skimage.transform.rescale(img,
                                    resize_ratio,
                                    multichannel=True,
                                    anti_aliasing=True,
                                    mode='constant',
                                    preserve_range=True)

    y_range = img.shape[0] - height
    x_range = img.shape[1] - width

    y_skip = random.randint(0, y_range)
    x_skip = random.randint(0, x_range)

    crop_img = img[y_skip:y_skip + height, x_skip:x_skip + width]

    return crop_img


def debug_data_generator(data_generator):

    print(f"Samples per batch : {len(data_generator)}")

    for batch_num in range(len(data_generator)):
        # for batch_num in range(2):
        #
        print(f"Batch {batch_num}")

        first_batch = data_generator[batch_num]

        input_data = first_batch[0]
        output_data = first_batch[1]

        def debug_nn_data(nn_data):
            for key, value in nn_data.items():
                print(f"Layer name {key}")
                print(f"Data shape {value.shape}")
                print(f"Data type {value.dtype}")

                for i in range(value.shape[0]):
                    multi = 1
                    if value[i].max() <= 1:
                        multi = 255
                    cv2.imwrite(f"debug/layer_{key}_{batch_num}_{i}.png",
                                value[i] * multi)

        debug_nn_data(input_data)
        debug_nn_data(output_data)

    # img, mask, orig = next(gen.get_sample())
    #
    # print(img.shape)
    # print(mask.shape)
    # print(orig.shape)
