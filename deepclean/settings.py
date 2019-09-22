import glob
from pathlib import Path

DETECTOR_NETWORK_BACKEND = "mobilenet"
DETECTOR_NETWORK_RESOLUTION = (512, 512)
DETECTOR_BATCH_SIZE = 4
DETECTOR_BATCHES_PER_EPOCH = 4000
DETECTOR_BATCHES_PER_EPOCH_VALIDATION = 100

DETECTOR_INPUT_LAYER_NAME = {
    'mobilenet': 'input_1',
    'mobilenetv2': 'input_1',
    'resnet18': 'data',
    'resnet34': 'data',
    'resnet152': 'data',
    'vgg16': 'input_1',
    'vgg19': 'input_1',
    'inceptionv3': 'input_1',
    'inceptionresnetv2': 'input_1',
}

REMOVER_VGG_WEIGHTS = "./training_weights/pytorch_to_keras_vgg16.h5"
# REMOVER_WEIGHTS = "./training_weights/weights.26-1.07.h5"
REMOVER_WEIGHTS = "./training_weights/remover-weights-improvement.hdf5"

IMAGE_TRAINING_DATA_DIR = Path("./data/train/")
IMAGE_VALIDATION_DATA_DIR = Path("./data/test/")

NEGATIVE_IMAGES = glob.glob('./background_images/*.JPG')