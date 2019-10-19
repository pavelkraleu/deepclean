import os
from keras.callbacks import ModelCheckpoint, CSVLogger
from deepclean import settings
from deepclean.remover_net.model_debugger import RemoverModelDebugger
from deepclean.remover_net.pconv_model import PConvUnet
from deepclean.remover_net.sequence_generator import SequenceGenerator


class RemoverNeuralNetwork:

    checkpoint_path = f'./training_weights_{settings.REMOVER_NETWORK_NAME}/weights-improvement.hdf5'
    csv_log_path_prefix = f'./training_weights_{settings.REMOVER_NETWORK_NAME}/training_log.'

    def __init__(self):
        os.makedirs(f'./training_weights_{settings.REMOVER_NETWORK_NAME}',
                    exist_ok=True)
        os.makedirs(f'./training_results_{settings.REMOVER_NETWORK_NAME}',
                    exist_ok=True)

        self.training_data = SequenceGenerator()
        self.validation_data = SequenceGenerator(validation=True)

        self.model = PConvUnet(vgg_weights=settings.REMOVER_VGG_WEIGHTS)

        print(f"Loading weights {self.checkpoint_path}")
        self.model.load(self.checkpoint_path, train_bn=False, lr=0.00005)
        self.model.model.summary(line_length=128)

    def fit(self):
        self.model.model.fit_generator(
            self.training_data,
            validation_data=self.validation_data,
            epochs=100,
            use_multiprocessing=True,
            verbose=1,
            callbacks=[
                CSVLogger(self.csv_log_path_prefix + 'complete.csv'),
                RemoverModelDebugger(self),
                ModelCheckpoint(
                    self.checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=True,
                )
            ],
        )
