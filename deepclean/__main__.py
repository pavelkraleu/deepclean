from deepclean.detector_net.detector_sequence_generator import DetectorSequenceGenerator
from deepclean.detector_net.network import SegmentationNeuralNetwork
from deepclean.remover_net.network import RemoverNeuralNetwork
from deepclean.cleaner_net.cleaner import Cleaner
from deepclean.tools import debug_data_generator

# net = SegmentationNeuralNetwork()
# net.fit()

# net = RemoverNeuralNetwork()
# net.fit()

#
# training_data = DetectorSequenceGenerator()
# debug_data_generator(training_data)

# debug_img = "/Users/me/dev/graffiti-dataset/dataset/test_set/square/IMG_3089.MOV_frames/thumb0028_IMG_3089.MOV.png"
# #
cleaner = Cleaner()
# # cleaner.process_frame(debug_img)
cleaner.process_video()
