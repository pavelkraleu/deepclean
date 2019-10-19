import keras
import matplotlib.pyplot as plt
from deepclean import settings


class RemoverModelDebugger(keras.callbacks.Callback):
    """
    This callback renders validation samples during training of remover network
    """
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_epoch_end(self, epoch, logs={}):
        for bi, batch in enumerate(self.trainer.validation_data):
            if bi % 100 != 0:
                continue
            result = self.trainer.model.model.predict(batch[0])
            for i, sample in enumerate(result):
                if i % 2 != 0:
                    continue

                self.save_sample(
                    batch[0]['inputs_img'][i], sample,
                    f'./training_results_{settings.REMOVER_NETWORK_NAME}/e{epoch}_b{bi}_s{i}.jpg'
                )

    def save_sample(self, x, y, output_path):
        fig, axs = plt.subplots(1, 2, figsize=(16, 9), dpi=150)
        axs[0].imshow(x)
        axs[1].imshow(y)

        for i in [0, 1]:
            axs[i].tick_params(left=False, bottom=False)
            axs[i].set_yticklabels([])
            axs[i].set_xticklabels([])

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
