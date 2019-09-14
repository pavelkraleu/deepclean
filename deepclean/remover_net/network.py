from keras.callbacks import ModelCheckpoint

from deepclean import settings
from deepclean.remover_net.remover_sequence_generator import RemoverSequenceGenerator
from deepclean.remover_net.pconv_model import PConvUnet


class RemoverNeuralNetwork:

    checkpoint_path = f'./training_weights/remover-weights-improvement.hdf5'

    def __init__(self):

        self.training_data = RemoverSequenceGenerator()
        self.validation_data = RemoverSequenceGenerator(validation=True)

        self.model = PConvUnet(vgg_weights=settings.REMOVER_VGG_WEIGHTS)

        print(f"Loading weights {settings.REMOVER_WEIGHTS}")
        self.model.load(settings.REMOVER_WEIGHTS, train_bn=False, lr=0.00005)
        self.model.model.summary(line_length=128)

    def fit(self):

        # RemoverModelDebugger(self).on_epoch_end(0)

        self.model.model.fit_generator(
            self.training_data,
            validation_data=self.validation_data,
            epochs=100,
            use_multiprocessing=True,
            verbose=1,
            callbacks=[
                ModelCheckpoint(
                    self.checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=True,
                )
            ],
        )