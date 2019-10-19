import glob
from pathlib import Path
from numpy.random import seed
from tensorflow.compat.v1 import set_random_seed
import random

random.seed(7)
seed(7)
set_random_seed(7)

DETECTOR_NETWORK_BACKEND = "resnet34"
DETECTOR_NETWORK_RESOLUTION = (512, 512)
DETECTOR_BATCH_SIZE = 12
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
REMOVER_NETWORK_NAME = "remover"
REMOVER_BATCH_SIZE = 4

IMAGE_TRAINING_DATA_DIR = Path("./data/train/")
IMAGE_VALIDATION_DATA_DIR = Path("./data/test/")

NEGATIVE_IMAGES = glob.glob('./background_images/*.JPG')

PREPARED_TRAINING_DATA_DIR = Path("./training_data/")
PREPARED_VALIDATION_DATA_DIR = Path("./training_data_validation/")

IMAGENET_DIR = Path("./imagenet/")

GENERATE_OUTPUT_THRESHOLD = 0.95

CLEANER_VIDEO_FILES_PATH = './video_files/*.MOV'
