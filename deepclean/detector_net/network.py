import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from segmentation_models.utils import set_trainable
from deepclean import settings
from deepclean.detector_net.model_debugger import SegmentationModelDebugger
from deepclean.detector_net.sequence_generator import SequenceGenerator


class SegmentationNeuralNetwork:

    checkpoint_path = f'./training_weights_{settings.DETECTOR_NETWORK_BACKEND}/weights-improvement.hdf5'
    csv_log_path_prefix = f'./training_weights_{settings.DETECTOR_NETWORK_BACKEND}/training_log.'

    def __init__(self):

        print(f"Network architecture {settings.DETECTOR_NETWORK_BACKEND}")

        os.makedirs(f'./training_weights_{settings.DETECTOR_NETWORK_BACKEND}',
                    exist_ok=True)
        os.makedirs(f'./training_results_{settings.DETECTOR_NETWORK_BACKEND}',
                    exist_ok=True)

        self.training_data = SequenceGenerator()
        self.validation_data = SequenceGenerator(validation=True)

        adam = Adam()

        self.model = Unet(settings.DETECTOR_NETWORK_BACKEND,
                          encoder_freeze=True,
                          input_shape=settings.DETECTOR_NETWORK_RESOLUTION +
                          (3, ))

        self.model.compile(adam, loss=bce_jaccard_loss, metrics=[iou_score])
        # self.model.summary(line_length=128)

        if os.path.exists(self.checkpoint_path):
            print(f'Loading weights from {self.checkpoint_path}')
            self.model.load_weights(self.checkpoint_path)

    def fit(self):
        if not os.path.exists(self.checkpoint_path):
            print('Training decoder')

            self.model.fit_generator(self.training_data,
                                     validation_data=self.validation_data,
                                     epochs=2,
                                     use_multiprocessing=True,
                                     workers=4,
                                     callbacks=[
                                         SegmentationModelDebugger(self, True),
                                         CSVLogger(self.csv_log_path_prefix +
                                                   'encoder.csv')
                                     ])

        set_trainable(self.model)

        self.model.fit_generator(
            self.training_data,
            validation_data=self.validation_data,
            epochs=50,
            use_multiprocessing=True,
            workers=4,
            callbacks=[
                SegmentationModelDebugger(self, False),
                ModelCheckpoint(self.checkpoint_path,
                                monitor='val_loss',
                                save_best_only=True,
                                mode='min'),
                ModelCheckpoint(self.checkpoint_path +
                                '.weights.{epoch:02d}-{loss:.2f}.hdf5',
                                monitor='val_loss',
                                save_best_only=True,
                                save_weights_only=True),
                CSVLogger(self.csv_log_path_prefix + 'complete.csv')
            ])
