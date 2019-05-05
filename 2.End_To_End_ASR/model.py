import os
import sys
sys.path.append("../../../")
import tensorflow as tf
import timit_parameter
from Utils.tensorflow_high_api import BasicUtil       #常规tensorflow的高级封装
from Utils.tensorflow_high_api import RnnUtil         #RNN系列的tensorflow的高级封装

class AcousticModel:
    def __init__(self):
        self.class_num=timit_parameter.CLASS_NUM
        self.basicUtil = BasicUtil()
        self.rnnUtil = RnnUtil()


    def forward(self):
        pass




class LanguageModel:
    pass


if __name__=="__main__":
    pass