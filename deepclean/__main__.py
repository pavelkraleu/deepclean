import glob
import os
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    from deepclean.cleaner_net.cleaner import Cleaner
    from deepclean.detector_net.network import SegmentationNeuralNetwork
    from deepclean.detector_net.sequence_generator import SequenceGenerator as DetectorSequenceGenerator
    from deepclean.remover_net.sequence_generator import SequenceGenerator as RemoverSequenceGenerator
    from deepclean.remover_net.network import RemoverNeuralNetwork
    from deepclean.tools import debug_data_generator
    from deepclean.training_data_generator.image_processor import ImageProcessor
    import click
    from deepclean import settings


def _remove_debug_directory():
    files = glob.glob("./debug/*")
    for f in files:
        os.remove(f)


@click.group()
def main():
    pass


@main.command(help="Train neural network detecting graffiti in image")
def train_segmentation_network():
    net = SegmentationNeuralNetwork()
    net.fit()


@main.command(help="Train neural network reconstructing original image")
def train_remover_network():
    net = RemoverNeuralNetwork()
    net.fit()


@main.command(help="Generate training samples")
@click.option('--num_samples',
              default=1000,
              help='Number of samples to generate',
              type=int)
def gen_training_data(num_samples):
    ImageProcessor(num_samples).run()


@main.command(help="Generate validation samples")
@click.option('--num_samples',
              default=1000,
              help='Number of samples to generate',
              type=int)
def get_validation_data(num_samples):
    ImageProcessor(num_samples, validation=True).run()


@main.command(help=f"Clean files in {settings.CLEANER_VIDEO_FILES_PATH}")
def clean_data():
    cleaner = Cleaner()
    cleaner.process_video()


@main.command(help="Writes training data for detector network into ./debug")
@click.option('--validation',
              default=False,
              help='Debug validation dataset',
              type=bool)
def debug_training_data(validation):
    _remove_debug_directory()
    training_data = DetectorSequenceGenerator(validation=validation)
    debug_data_generator(training_data)


@main.command(help="Writes training data for remover network into ./debug")
@click.option('--validation',
              default=False,
              help='Debug validation dataset',
              type=bool)
def debug_validation_data(validation):
    _remove_debug_directory()
    training_data = RemoverSequenceGenerator(validation=validation)
    debug_data_generator(training_data)


main()
