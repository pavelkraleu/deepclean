import numpy as np
import cv2


def paste_on_background(image, mask, background_image):
    """
    Paste given image on background image with given mask
    """
    foreground = image.astype(float)
    background = background_image.astype(float)

    alpha = cv2.inRange(mask, 0, 0)
    alpha = cv2.bitwise_not(alpha).astype(float) / 255
    alpha = np.stack((alpha, ) * 3, axis=-1)

    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)

    return cv2.add(foreground, background).astype(np.uint8)


def debug_data_generator(data_generator):
    """
    Iterates over given Keras sequence generator and writes samples to ./debug
    """
    print(f"Number of batches : {len(data_generator)}")

    for batch_num in range(len(data_generator)):
        print(f"Batch number {batch_num}")
        first_batch = data_generator[batch_num]
        input_data = first_batch[0]
        output_data = first_batch[1]

        def debug_nn_data(nn_data):
            for key, value in nn_data.items():
                print(f"Layer name : {key}")
                print(f"Data shape : {value.shape}")
                print(f"Data type : {value.dtype}")
                print()

                for i in range(value.shape[0]):
                    multi = 1
                    if value[i].max() <= 1:
                        multi = 255
                    cv2.imwrite(f"debug/layer_{key}_{batch_num}_{i}.png",
                                value[i] * multi)

        debug_nn_data(input_data)
        debug_nn_data(output_data)
