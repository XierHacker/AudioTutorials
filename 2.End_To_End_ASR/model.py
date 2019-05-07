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
        self.hidden_units_num=256
        self.basicUtil = BasicUtil()
        self.rnnUtil = RnnUtil()

    def forward(self,mfcc,mfcc_len,keep_prob,reuse=False):
        # forward part
        lstm_forward1 = self.rnnUtil.createLSTMCell(self.hidden_units_num, keep_prob, "lstm_forward1", reuse)
        lstm_forward2 = self.rnnUtil.createLSTMCell(self.hidden_units_num, keep_prob, "lstm_forward2", reuse)
        lstm_forward3 = self.rnnUtil.createLSTMCell(self.hidden_units_num, keep_prob, "lstm_forward3", reuse)
        lstm_forward4 = self.rnnUtil.createLSTMCell(self.hidden_units_num, keep_prob, "lstm_forward4", reuse)
        lstm_forward = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_forward1, lstm_forward2,lstm_forward3,lstm_forward4])

        # backward part
        lstm_backward1 = self.rnnUtil.createLSTMCell(self.hidden_units_num, keep_prob, "lstm_backward1", reuse)
        lstm_backward2 = self.rnnUtil.createLSTMCell(self.hidden_units_num, keep_prob, "lstm_backward2", reuse)
        lstm_backward3 = self.rnnUtil.createLSTMCell(self.hidden_units_num, keep_prob, "lstm_backward3", reuse)
        lstm_backward4 = self.rnnUtil.createLSTMCell(self.hidden_units_num, keep_prob, "lstm_backward4", reuse)
        lstm_backward = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_backward1, lstm_backward2,lstm_backward3,lstm_backward4])

        # LSTM编码过程
        # encoder_outputs是拼接之后的输出，这里暂时用不到encoder_states
        encoder_outputs, encoder_states = self.rnnUtil.LSTM_Encoder(
            inputs=mfcc,
            cell_forward=lstm_forward,
            cell_backward=lstm_backward,
            seq_len=mfcc_len,
            scope_name="encoder"
        )

        # [batch_size*max_time, cell_fw.output_size*2]
        #h = tf.reshape(encoder_outputs, [-1, 2 * self.hidden_units_num], "h_reshaped")

        # fully connect layer
        logits = self.basicUtil.linear(
            inputs=encoder_outputs,
            units=self.class_num,
            activation=None,
            regularizer=None,
            keep_rate=keep_prob,
            name="logits",
            reuse=reuse
        )
        return logits  # shape of logits:[batch_size*max_time, 29]


class LanguageModel:
    pass


if __name__=="__main__":
    pass