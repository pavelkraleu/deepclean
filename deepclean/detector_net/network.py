import os
from keras.callbacks import ModelCheckpoint
from keras.layers import GaussianNoise
from keras.optimizers import Adam
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from segmentation_models.utils import set_trainable
from deepclean import settings
from deepclean.detector_net.detector_sequence_generator import DetectorSequenceGenerator
from deepclean.detector_net.model_debugger import SegmentationModelDebugger


class SegmentationNeuralNetwork:

    checkpoint_path = f'./training_weights/weights-improvement.hdf5'

    def __init__(self):

        self.training_data = DetectorSequenceGenerator()
        self.validation_data = DetectorSequenceGenerator(validation=True)

        adam = Adam()

        self.model = Unet(settings.DETECTOR_NETWORK_BACKEND,
                          encoder_freeze=True,
                          input_shape=settings.DETECTOR_NETWORK_RESOLUTION +
                          (3, ))

        self.model.compile(adam, loss=bce_jaccard_loss, metrics=[iou_score])
        self.model.summary(line_length=128)

        if os.path.exists(self.checkpoint_path):
            print(f'Loading weights from {self.checkpoint_path}')
            self.model.load_weights(self.checkpoint_path)

            # SegmentationModelDebugger(self).debug_validation_data(0)
            # SegmentationModelDebugger(self).debug_video_data(0)

    def fit(self):
        # for i in range(3500, len(self.training_data)):
        #     print(f"generating {i}")
        #     img = self.training_data[i]

        if not os.path.exists(self.checkpoint_path):
            print('Training decoder')

            self.model.fit_generator(
                self.training_data,
                validation_data=self.validation_data,
                epochs=6,
                use_multiprocessing=True,
                workers=4,
                callbacks=[SegmentationModelDebugger(self)])

        set_trainable(self.model)

        self.model.fit_generator(self.training_data,
                                 validation_data=self.validation_data,
                                 epochs=41,
                                 use_multiprocessing=True,
                                 workers=4,
                                 callbacks=[
                                     SegmentationModelDebugger(self),
                                     ModelCheckpoint(self.checkpoint_path,
                                                     monitor='val_loss',
                                                     verbose=False,
                                                     save_best_only=True,
                                                     mode='min')
                                 ])
